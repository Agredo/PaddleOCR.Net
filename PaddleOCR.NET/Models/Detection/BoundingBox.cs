namespace PaddleOCR.NET.Models.Detection;

/// <summary>
/// Bounding box for detected text
/// </summary>
public class BoundingBox
{
    /// <summary>
    /// The four corner points of the box (clockwise, starting from top-left)
    /// </summary>
    public (float X, float Y)[] Points { get; init; } = Array.Empty<(float, float)>();
    
    /// <summary>
    /// Confidence score (0-1)
    /// </summary>
    public float Confidence { get; init; }
    
    /// <summary>
    /// Creates a new bounding box
    /// </summary>
    /// <param name="points">Four corner points with 2 coordinates each</param>
    /// <param name="confidence">Confidence score</param>
    public BoundingBox(float[][] points, float confidence)
    {
 if (points.Length != 4 || points.Any(p => p.Length != 2))
     throw new ArgumentException("BoundingBox requires exactly 4 points with 2 coordinates each");
        
        Points = points.Select(p => (p[0], p[1])).ToArray();
        Confidence = confidence;
    }
}
