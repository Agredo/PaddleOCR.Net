namespace PaddleOCR.NET.Models.Detection;

/// <summary>
/// Interface for text detection models (PP-OCR Detection)
/// </summary>
public interface IDetectionModel : IDisposable
{
    /// <summary>
    /// Performs text detection on an image
    /// </summary>
    /// <param name="imageData">Image data as byte array</param>
    /// <returns>Detection result with bounding boxes</returns>
    DetectionResult Detect(byte[] imageData);

    /// <summary>
    /// Performs text detection on multiple images
    /// </summary>
    /// <param name="imageBatch">Array of image data</param>
    /// <returns>List of detection results</returns>
    IEnumerable<DetectionResult> DetectBatch(byte[][] imageBatch);
}
