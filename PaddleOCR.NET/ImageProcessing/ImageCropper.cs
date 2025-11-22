using SkiaSharp;
using PaddleOCR.NET.Models.Detection;

namespace PaddleOCR.NET.ImageProcessing;

/// <summary>
/// Handles image cropping for OCR regions
/// </summary>
public static class ImageCropper
{
    /// <summary>
    /// Crops a region from an image using a bounding box
    /// </summary>
    /// <param name="bitmap">Source bitmap</param>
    /// <param name="box">Bounding box defining the region to crop</param>
    /// <returns>Cropped bitmap</returns>
    public static SKBitmap CropRegion(SKBitmap bitmap, BoundingBox box)
    {
        if (bitmap == null)
            throw new ArgumentNullException(nameof(bitmap));
        if (box == null)
            throw new ArgumentNullException(nameof(box));
        
        // Get the bounding rectangle
        var rect = GetBoundingRect(box);
        
        // Ensure the rectangle is within image bounds
        var clampedRect = ClampRect(rect, bitmap.Width, bitmap.Height);
        
        // Create a new bitmap for the cropped region
        var croppedBitmap = new SKBitmap(clampedRect.Width, clampedRect.Height);
        
        using var canvas = new SKCanvas(croppedBitmap);
        
        // Draw the cropped region
        var sourceRect = new SKRect(clampedRect.Left, clampedRect.Top, 
                                    clampedRect.Right, clampedRect.Bottom);
        var destRect = new SKRect(0, 0, clampedRect.Width, clampedRect.Height);
        
        canvas.DrawBitmap(bitmap, sourceRect, destRect);
        
        return croppedBitmap;
    }
    
    /// <summary>
    /// Crops regions from an image using multiple bounding boxes
    /// </summary>
    /// <param name="bitmap">Source bitmap</param>
    /// <param name="boxes">List of bounding boxes</param>
    /// <returns>Array of cropped bitmaps</returns>
    public static SKBitmap[] CropRegions(SKBitmap bitmap, IReadOnlyList<BoundingBox> boxes)
    {
        if (bitmap == null)
            throw new ArgumentNullException(nameof(bitmap));
        if (boxes == null)
            throw new ArgumentNullException(nameof(boxes));
        
        var croppedBitmaps = new SKBitmap[boxes.Count];
        
        for (int i = 0; i < boxes.Count; i++)
        {
            croppedBitmaps[i] = CropRegion(bitmap, boxes[i]);
        }
        
        return croppedBitmaps;
    }
    
    /// <summary>
    /// Crops a region from image bytes using a bounding box
    /// </summary>
    /// <param name="imageData">Source image data</param>
    /// <param name="box">Bounding box defining the region to crop</param>
    /// <returns>Cropped image as byte array (PNG format)</returns>
    public static byte[] CropRegion(byte[] imageData, BoundingBox box)
    {
        using var bitmap = ImageLoader.LoadWithOrientation(imageData);
        if (bitmap == null)
            throw new InvalidOperationException("Failed to load image");
        
        using var croppedBitmap = CropRegion(bitmap, box);
        
        // Encode to PNG
        using var image = SKImage.FromBitmap(croppedBitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        
        return data.ToArray();
    }
    
    /// <summary>
    /// Gets the axis-aligned bounding rectangle from a bounding box
    /// </summary>
    private static SKRectI GetBoundingRect(BoundingBox box)
    {
        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        
        foreach (var point in box.Points)
        {
            minX = Math.Min(minX, point.X);
            minY = Math.Min(minY, point.Y);
            maxX = Math.Max(maxX, point.X);
            maxY = Math.Max(maxY, point.Y);
        }
        
        return new SKRectI(
            (int)Math.Floor(minX),
            (int)Math.Floor(minY),
            (int)Math.Ceiling(maxX),
            (int)Math.Ceiling(maxY)
        );
    }
    
    /// <summary>
    /// Clamps a rectangle to be within image bounds
    /// </summary>
    private static SKRectI ClampRect(SKRectI rect, int imageWidth, int imageHeight)
    {
        int left = Math.Max(0, rect.Left);
        int top = Math.Max(0, rect.Top);
        int right = Math.Min(imageWidth, rect.Right);
        int bottom = Math.Min(imageHeight, rect.Bottom);
        
        // Ensure we have a valid rectangle
        if (right <= left) right = left + 1;
        if (bottom <= top) bottom = top + 1;
        
        return new SKRectI(left, top, right, bottom);
    }
}
