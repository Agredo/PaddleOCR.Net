using SkiaSharp;

namespace PaddleOCR.NET.ImageProcessing;

/// <summary>
/// Handles image resizing for PP-OCR models
/// </summary>
public static class ImageResizer
{
    /// <summary>
    /// Calculates the target size for image resizing while maintaining aspect ratio.
    /// The longer side will be resized to targetSize.
    /// </summary>
    /// <param name="width">Original width</param>
    /// <param name="height">Original height</param>
    /// <param name="targetSize">Target size for the longer side (default: 960)</param>
    /// <returns>Calculated width and height</returns>
    public static (int Width, int Height) CalculateResizeSize(int width, int height, int targetSize = 960)
    {
        if (width > height)
        {
            var ratio = (float)targetSize / width;
            return (targetSize, (int)(height * ratio));
        }
        else
        {
            var ratio = (float)targetSize / height;
            return ((int)(width * ratio), targetSize);
        }
    }
    
    /// <summary>
    /// Calculates padded size to ensure dimensions are multiples of 32 (required for PP-OCR)
    /// </summary>
    /// <param name="width">Original width</param>
    /// <param name="height">Original height</param>
    /// <returns>Padded width and height</returns>
    public static (int Width, int Height) CalculatePaddedSize(int width, int height)
    {
        return (
            ((width + 31) / 32) * 32,
            ((height + 31) / 32) * 32
        );
    }
    
    /// <summary>
    /// Resizes a bitmap while maintaining aspect ratio
    /// </summary>
    /// <param name="bitmap">Source bitmap</param>
    /// <param name="targetSize">Target size for the longer side</param>
    /// <returns>Resized bitmap</returns>
    public static SKBitmap Resize(SKBitmap bitmap, int targetSize = 960)
    {
        var (width, height) = CalculateResizeSize(bitmap.Width, bitmap.Height, targetSize);

        var resized = bitmap.Resize(
            new SKImageInfo(width, height), 
            new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
        
        if (resized == null)
            throw new InvalidOperationException("Failed to resize bitmap");

        return resized;
    }
}
