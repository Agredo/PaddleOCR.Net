using SkiaSharp;

namespace PaddleOCR.NET.Models.Recognition;

/// <summary>
/// Interface for text recognition models (PP-OCR Recognition)
/// </summary>
public interface IRecognitionModel : IDisposable
{
    /// <summary>
    /// Recognizes text in a single image
    /// </summary>
    /// <param name="imageData">Image data as byte array</param>
    /// <returns>Recognized text with confidence</returns>
    RecognizedText Recognize(byte[] imageData);
    
    /// <summary>
    /// Recognizes text in multiple images (batch processing)
    /// </summary>
    /// <param name="imageBatch">Array of image data</param>
    /// <returns>Recognition result with all texts</returns>
    RecognitionResult RecognizeBatch(byte[][] imageBatch);
    
    /// <summary>
    /// Recognizes text in multiple bitmaps (batch processing)
    /// </summary>
    /// <param name="bitmaps">Array of SKBitmap objects</param>
    /// <returns>Recognition result with all texts</returns>
    RecognitionResult RecognizeBatch(SKBitmap[] bitmaps);
}
