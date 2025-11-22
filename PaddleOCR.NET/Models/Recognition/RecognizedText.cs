namespace PaddleOCR.NET.Models.Recognition;

/// <summary>
/// Contains recognized text from a single image or text region
/// </summary>
public class RecognizedText
{
    /// <summary>
    /// The recognized text string
    /// </summary>
    public string Text { get; init; } = string.Empty;
    
    /// <summary>
    /// Overall confidence score (0-1)
    /// </summary>
    public float Confidence { get; init; }
    
    /// <summary>
    /// Per-character confidence scores (optional)
    /// </summary>
    public float[]? CharConfidences { get; init; }
    
    /// <summary>
    /// Creates a new recognized text result
    /// </summary>
    /// <param name="text">Recognized text</param>
    /// <param name="confidence">Overall confidence score</param>
    /// <param name="charConfidences">Optional per-character confidence scores</param>
    public RecognizedText(string text, float confidence, float[]? charConfidences = null)
    {
        Text = text;
        Confidence = confidence;
        CharConfidences = charConfidences;
    }
}
