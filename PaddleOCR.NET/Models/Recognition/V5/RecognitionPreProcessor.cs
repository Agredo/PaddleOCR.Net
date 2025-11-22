using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace PaddleOCR.NET.Models.Recognition.V5;

/// <summary>
/// Preprocessing for PP-OCR Recognition models
/// </summary>
public static class RecognitionPreProcessor
{
    /// <summary>
    /// Fixed height for recognition model input
    /// </summary>
    public const int TargetHeight = 48;
    
    /// <summary>
    /// Maximum width for recognition model input
    /// </summary>
    public const int MaxWidth = 320;
    
    /// <summary>
    /// Preprocesses a batch of bitmaps for recognition
    /// </summary>
    /// <param name="bitmaps">Array of source bitmaps</param>
    /// <returns>Normalized tensor with shape [batch, 3, 48, 320]</returns>
    public static DenseTensor<float> PreprocessBatch(SKBitmap[] bitmaps)
    {
        int batchSize = bitmaps.Length;
        var tensor = new DenseTensor<float>([batchSize, 3, TargetHeight, MaxWidth]);
        
        for (int b = 0; b < batchSize; b++)
        {
            PreprocessSingle(bitmaps[b], tensor, b);
        }
        
        return tensor;
    }
    
    /// <summary>
    /// Preprocesses a single bitmap into a batch tensor at specified index
    /// </summary>
    /// <param name="bitmap">Source bitmap</param>
    /// <param name="tensor">Target tensor</param>
    /// <param name="batchIndex">Index in the batch</param>
    private static void PreprocessSingle(SKBitmap bitmap, DenseTensor<float> tensor, int batchIndex)
    {
        int originalWidth = bitmap.Width;
        int originalHeight = bitmap.Height;
        
        // Calculate aspect ratio and target width
        float ratio = (float)originalWidth / originalHeight;
        int resizedWidth = (int)Math.Ceiling(TargetHeight * ratio);
        
        // Cap at max width
        if (resizedWidth > MaxWidth)
        {
            resizedWidth = MaxWidth;
        }
        
        // Resize image to (resizedWidth, TargetHeight)
        using var resizedBitmap = bitmap.Resize(
            new SKImageInfo(resizedWidth, TargetHeight),
            new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
        
        if (resizedBitmap == null)
            throw new InvalidOperationException("Failed to resize bitmap");
        
        // Get pixel data
        var pixels = resizedBitmap.Pixels;
        
        // Normalize and fill tensor: (pixel / 255 - 0.5) / 0.5
        for (int y = 0; y < TargetHeight; y++)
        {
            for (int x = 0; x < resizedWidth; x++)
            {
                var pixel = pixels[y * resizedWidth + x];
                
                // Extract RGB components (SkiaSharp uses BGRA format)
                var r = pixel.Red;
                var g = pixel.Green;
                var b = pixel.Blue;
                
                // Apply normalization: (pixel / 255 - 0.5) / 0.5
                tensor[batchIndex, 0, y, x] = (r / 255f - 0.5f) / 0.5f;
                tensor[batchIndex, 1, y, x] = (g / 255f - 0.5f) / 0.5f;
                tensor[batchIndex, 2, y, x] = (b / 255f - 0.5f) / 0.5f;
            }
        }
        
        // Right-side padding remains 0 (already initialized by tensor constructor)
        // Padding area: [resizedWidth, MaxWidth)
    }
}
