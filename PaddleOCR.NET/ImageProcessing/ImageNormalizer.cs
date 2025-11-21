using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace PaddleOCR.NET.ImageProcessing;

/// <summary>
/// Handles image normalization for PP-OCR models
/// </summary>
public static class ImageNormalizer
{
    /// <summary>
    /// ImageNet normalization mean values (RGB)
    /// </summary>
    public static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    
    /// <summary>
    /// ImageNet normalization standard deviation values (RGB)
    /// </summary>
    public static readonly float[] Std = [0.229f, 0.224f, 0.225f];
    
    /// <summary>
    /// Normalizes a bitmap to a tensor with PP-OCR normalization.
    /// Formula: (pixel / 255 - mean) / std
    /// </summary>
    /// <param name="bitmap">Source bitmap</param>
    /// <param name="paddedWidth">Target tensor width (with padding)</param>
    /// <param name="paddedHeight">Target tensor height (with padding)</param>
    /// <returns>Normalized tensor with shape [1, 3, height, width]</returns>
    public static DenseTensor<float> NormalizeToTensor(
        SKBitmap bitmap, 
        int paddedWidth, 
        int paddedHeight)
    {
        var tensor = new DenseTensor<float>([1, 3, paddedHeight, paddedWidth]);

        // Get pixel data
        var pixels = bitmap.Pixels;
        
        for (int y = 0; y < bitmap.Height; y++)
        {
            for (int x = 0; x < bitmap.Width; x++)
            {
                var pixel = pixels[y * bitmap.Width + x];
                
                // Extract RGB components (SkiaSharp uses BGRA format)
                var r = pixel.Red;
                var g = pixel.Green;
                var b = pixel.Blue;
                
                // RGB -> normalized tensor data
                tensor[0, 0, y, x] = (r / 255f - Mean[0]) / Std[0];
                tensor[0, 1, y, x] = (g / 255f - Mean[1]) / Std[1];
                tensor[0, 2, y, x] = (b / 255f - Mean[2]) / Std[2];
            }
        }

        // Padding areas remain 0 (already initialized by tensor constructor)
        return tensor;
    }
}
