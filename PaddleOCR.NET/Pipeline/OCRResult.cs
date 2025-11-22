using PaddleOCR.NET.Models.Detection;
using PaddleOCR.NET.Models.Recognition;

namespace PaddleOCR.NET.Pipeline;

/// <summary>
/// Result of OCR processing containing both detection and recognition results
/// </summary>
public class OCRResult
{
    /// <summary>
    /// Individual text regions with their locations and recognized text
    /// </summary>
    public IReadOnlyList<OCRTextRegion> TextRegions { get; init; } = Array.Empty<OCRTextRegion>();
    
    /// <summary>
    /// Original detection result
    /// </summary>
    public DetectionResult DetectionResult { get; init; } = null!;
    
    /// <summary>
    /// Total number of text regions found
    /// </summary>
    public int Count => TextRegions.Count;
    
    /// <summary>
    /// Gets all recognized text concatenated
    /// </summary>
    /// <param name="separator">Separator between text regions (default: newline)</param>
    /// <returns>All text combined</returns>
    public string GetFullText(string separator = "\n")
    {
        return string.Join(separator, TextRegions.Select(r => r.Text.Text));
    }
}

/// <summary>
/// Represents a single text region with location and recognized text
/// </summary>
public class OCRTextRegion
{
    /// <summary>
    /// Bounding box of the text region
    /// </summary>
    public BoundingBox BoundingBox { get; init; } = null!;
    
    /// <summary>
    /// Recognized text with confidence
    /// </summary>
    public RecognizedText Text { get; init; } = null!;
    
    /// <summary>
    /// Index of this region in the detection results
    /// </summary>
    public int Index { get; init; }
}
