namespace PaddleOCR.NET.Models.Detection;

/// <summary>
/// Result of text detection
/// </summary>
public class DetectionResult
{
    /// <summary>
    /// List of detected bounding boxes
    /// </summary>
    public IReadOnlyList<BoundingBox> Boxes { get; init; } = Array.Empty<BoundingBox>();
    
    /// <summary>
    /// Processed image size (after padding)
    /// </summary>
    public (int Width, int Height) ProcessedImageSize { get; init; }
    
    /// <summary>
    /// Original image size
    /// </summary>
    public (int Width, int Height) OriginalImageSize { get; init; }
}
