namespace PaddleOCR.NET.Models.Detection.V5;

/// <summary>
/// Fluent builder for DetectionModelV5
/// </summary>
public class DetectionModelV5Builder
{
    private string? modelPath;
    private int targetSize = 960;
    private float threshold = 0.15f;
    private float boxThreshold = 0.3f;
    private float unclipRatio = 1.6f;
    private bool mergeBoxes = false;
    private float mergeDistanceThreshold = 0.5f;
    private float mergeOverlapThreshold = 0.1f;

    /// <summary>
    /// Sets the path to the model file
    /// </summary>
    /// <param name="path">Path to det.onnx</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithModelPath(string path)
    {
        modelPath = path;
        return this;
    }

    /// <summary>
    /// Sets the target size for the longer image side
    /// </summary>
    /// <param name="size">Target size (must be greater than 0)</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithTargetSize(int size)
    {
        if (size <= 0)
            throw new ArgumentException("TargetSize must be greater than 0", nameof(size));
        
        targetSize = size;
        return this;
    }

    /// <summary>
    /// Sets the detection threshold
    /// </summary>
    /// <param name="value">Threshold value (0-1)</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithThreshold(float value)
    {
        if (value < 0 || value > 1)
            throw new ArgumentException("Threshold must be between 0 and 1", nameof(value));
        
        threshold = value;
        return this;
    }

    /// <summary>
    /// Sets the bounding box threshold
    /// </summary>
    /// <param name="value">Box threshold value (0-1)</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithBoxThreshold(float value)
    {
        if (value < 0 || value > 1)
            throw new ArgumentException("BoxThreshold must be between 0 and 1", nameof(value));
        
        boxThreshold = value;
        return this;
    }

    /// <summary>
    /// Sets the unclip ratio for expanding detected boxes
    /// </summary>
    /// <param name="ratio">Unclip ratio</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithUnclipRatio(float ratio)
    {
        unclipRatio = ratio;
        return this;
    }

    /// <summary>
    /// Enables box merging for overlapping or nearby boxes
    /// </summary>
    /// <param name="enable">True to enable box merging</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithBoxMerging(bool enable = true)
    {
        mergeBoxes = enable;
        return this;
    }

    /// <summary>
    /// Sets the distance threshold for box merging
    /// </summary>
    /// <param name="threshold">Maximum distance between boxes to merge, as ratio of box height (default: 0.5)</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithMergeDistanceThreshold(float threshold)
    {
        if (threshold < 0)
            throw new ArgumentException("MergeDistanceThreshold must be non-negative", nameof(threshold));
        
        mergeDistanceThreshold = threshold;
        return this;
    }

    /// <summary>
    /// Sets the overlap threshold for box merging
    /// </summary>
    /// <param name="threshold">Minimum IOU for merging overlapping boxes (default: 0.1)</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public DetectionModelV5Builder WithMergeOverlapThreshold(float threshold)
    {
        if (threshold < 0 || threshold > 1)
            throw new ArgumentException("MergeOverlapThreshold must be between 0 and 1", nameof(threshold));
        
        mergeOverlapThreshold = threshold;
        return this;
    }

    /// <summary>
    /// Builds the DetectionModelV5 instance
    /// </summary>
    /// <returns>Configured DetectionModelV5 instance</returns>
    /// <exception cref="InvalidOperationException">Thrown when ModelPath is not set</exception>
    public DetectionModelV5 Build()
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new InvalidOperationException("ModelPath must be set before building");

        return new DetectionModelV5(
            modelPath, 
            targetSize, 
            threshold, 
            boxThreshold, 
            unclipRatio,
            mergeBoxes,
            mergeDistanceThreshold,
            mergeOverlapThreshold);
    }
}
