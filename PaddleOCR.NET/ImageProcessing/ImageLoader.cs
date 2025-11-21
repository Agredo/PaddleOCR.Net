using SkiaSharp;

namespace PaddleOCR.NET.ImageProcessing;

/// <summary>
/// Handles image loading with automatic EXIF orientation correction
/// </summary>
public static class ImageLoader
{
    /// <summary>
    /// Loads an image from byte array and applies EXIF orientation correction
    /// </summary>
    /// <param name="imageData">Raw image data</param>
    /// <returns>Bitmap with corrected orientation</returns>
    public static SKBitmap LoadWithOrientation(byte[] imageData)
    {
        using var stream = new MemoryStream(imageData);
        return LoadWithOrientation(stream);
    }

    /// <summary>
    /// Loads an image from a file path and applies EXIF orientation correction
    /// </summary>
    /// <param name="imagePath">Path to the image file</param>
    /// <returns>Bitmap with corrected orientation</returns>
    public static SKBitmap LoadWithOrientation(string imagePath)
    {
        using var stream = File.OpenRead(imagePath);
        return LoadWithOrientation(stream);
    }

    /// <summary>
    /// Loads an image from a stream and applies EXIF orientation correction
    /// </summary>
    /// <param name="stream">Image stream</param>
    /// <returns>Bitmap with corrected orientation</returns>
    public static SKBitmap LoadWithOrientation(Stream stream)
    {
        // Use SKCodec to detect the encoded origin (EXIF orientation)
        using var codec = SKCodec.Create(stream);
        if (codec == null)
            throw new InvalidOperationException("Failed to create codec from image stream");

        var origin = codec.EncodedOrigin;
        
        Console.WriteLine($"[IMAGE-LOADER] Detected EXIF orientation: {origin}");

        // Decode the image
        var info = codec.Info;
        var bitmap = new SKBitmap(info);
        var result = codec.GetPixels(bitmap.Info, bitmap.GetPixels());
        
        if (result != SKCodecResult.Success)
            throw new InvalidOperationException($"Failed to decode image: {result}");

        // Apply orientation correction if needed
        if (origin != SKEncodedOrigin.TopLeft)
        {
            Console.WriteLine($"[IMAGE-LOADER] Applying orientation correction: {origin} -> TopLeft");
            var correctedBitmap = ApplyOrientation(bitmap, origin);
            bitmap.Dispose();
            return correctedBitmap;
        }

        Console.WriteLine("[IMAGE-LOADER] No orientation correction needed");
        return bitmap;
    }

    /// <summary>
    /// Applies EXIF orientation transformation to a bitmap
    /// </summary>
    /// <param name="bitmap">Source bitmap</param>
    /// <param name="origin">EXIF orientation</param>
    /// <returns>Transformed bitmap</returns>
    private static SKBitmap ApplyOrientation(SKBitmap bitmap, SKEncodedOrigin origin)
    {
        // Determine if dimensions need to be swapped (for 90/270 degree rotations)
        var requiresSwap = origin == SKEncodedOrigin.LeftTop ||
                          origin == SKEncodedOrigin.RightTop ||
                          origin == SKEncodedOrigin.RightBottom ||
                          origin == SKEncodedOrigin.LeftBottom;

        var targetWidth = requiresSwap ? bitmap.Height : bitmap.Width;
        var targetHeight = requiresSwap ? bitmap.Width : bitmap.Height;

        var targetInfo = new SKImageInfo(targetWidth, targetHeight, bitmap.ColorType, bitmap.AlphaType);
        var targetBitmap = new SKBitmap(targetInfo);

        using var canvas = new SKCanvas(targetBitmap);
        
        // Apply transformation based on EXIF orientation
        switch (origin)
        {
            case SKEncodedOrigin.TopRight:
                // Flip horizontal
                canvas.Scale(-1, 1);
                canvas.Translate(-targetWidth, 0);
                break;

            case SKEncodedOrigin.BottomRight:
                // Rotate 180 degrees
                canvas.RotateDegrees(180, targetWidth / 2f, targetHeight / 2f);
                break;

            case SKEncodedOrigin.BottomLeft:
                // Flip vertical
                canvas.Scale(1, -1);
                canvas.Translate(0, -targetHeight);
                break;

            case SKEncodedOrigin.LeftTop:
                // Rotate 90 degrees CCW and flip horizontal
                canvas.Translate(0, targetHeight);
                canvas.RotateDegrees(-90);
                canvas.Scale(-1, 1);
                canvas.Translate(-bitmap.Width, 0);
                break;

            case SKEncodedOrigin.RightTop:
                // Rotate 90 degrees CW
                canvas.Translate(targetWidth, 0);
                canvas.RotateDegrees(90);
                break;

            case SKEncodedOrigin.RightBottom:
                // Rotate 90 degrees CW and flip horizontal
                canvas.Translate(targetWidth, 0);
                canvas.RotateDegrees(90);
                canvas.Scale(-1, 1);
                canvas.Translate(-bitmap.Width, 0);
                break;

            case SKEncodedOrigin.LeftBottom:
                // Rotate 90 degrees CCW
                canvas.Translate(0, targetHeight);
                canvas.RotateDegrees(-90);
                break;

            case SKEncodedOrigin.TopLeft:
            default:
                // No transformation needed
                break;
        }

        canvas.DrawBitmap(bitmap, 0, 0);
        
        return targetBitmap;
    }
}
