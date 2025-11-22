namespace PaddleOCR.NET.Models.Recognition.V5;

/// <summary>
/// Fluent builder for RecognitionModelV5
/// </summary>
public class RecognitionModelV5Builder
{
    private string? modelPath;
    private string? characterDictPath;
    private int batchSize = 6;

    /// <summary>
    /// Sets the path to the recognition model file
    /// </summary>
    /// <param name="path">Path to rec.onnx</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public RecognitionModelV5Builder WithModelPath(string path)
    {
        modelPath = path;
        return this;
    }

    /// <summary>
    /// Sets the path to the character dictionary file
    /// </summary>
    /// <param name="path">Path to character dictionary (e.g., ppocr_keys_v1.txt)</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public RecognitionModelV5Builder WithCharacterDict(string path)
    {
        characterDictPath = path;
        return this;
    }

    /// <summary>
    /// Sets the batch size for processing multiple images
    /// </summary>
    /// <param name="size">Batch size (must be greater than 0)</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public RecognitionModelV5Builder WithBatchSize(int size)
    {
        if (size <= 0)
            throw new ArgumentException("Batch size must be greater than 0", nameof(size));
        
        batchSize = size;
        return this;
    }

    /// <summary>
    /// Builds the RecognitionModelV5 instance
    /// </summary>
    /// <returns>Configured RecognitionModelV5 instance</returns>
    /// <exception cref="InvalidOperationException">Thrown when required parameters are not set</exception>
    public RecognitionModelV5 Build()
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new InvalidOperationException("ModelPath must be set before building");
        
        if (string.IsNullOrEmpty(characterDictPath))
            throw new InvalidOperationException("CharacterDict path must be set before building");

        return new RecognitionModelV5(modelPath, characterDictPath, batchSize);
    }
}
