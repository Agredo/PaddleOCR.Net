namespace PaddleOCR.NET.Models.Recognition;

/// <summary>
/// Result of batch text recognition
/// </summary>
public class RecognitionResult
{
    /// <summary>
    /// List of recognized texts from batch processing
    /// </summary>
    public IReadOnlyList<RecognizedText> Texts { get; init; } = Array.Empty<RecognizedText>();
    
    /// <summary>
    /// Number of images processed
    /// </summary>
    public int Count => Texts.Count;
}
