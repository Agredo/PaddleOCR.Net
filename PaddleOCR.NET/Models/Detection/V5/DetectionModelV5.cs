using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PaddleOCR.NET.ImageProcessing;
using PaddleOCR.NET.Tensor;
using SkiaSharp;

namespace PaddleOCR.NET.Models.Detection.V5;

/// <summary>
/// PP-OCRv5 Detection Model Implementation
/// </summary>
public class DetectionModelV5 : IDetectionModel
{
    private readonly InferenceSession session;
    private readonly int targetSize;
    private readonly float threshold;
    private readonly float boxThreshold;
    private readonly float unclipRatio;
    private bool disposed;

    /// <summary>
    /// Creates a new instance of the PP-OCRv5 Detection Model
    /// </summary>
    /// <param name="modelPath">Path to the det.onnx file</param>
    /// <param name="targetSize">Target size for the longer image side (default: 960)</param>
    /// <param name="threshold">Detection threshold (default: 0.15)</param>
    /// <param name="boxThreshold">Bounding box threshold (default: 0.3)</param>
    /// <param name="unclipRatio">Ratio for expanding boxes (default: 1.6)</param>
    public DetectionModelV5(
        string modelPath, 
        int targetSize = 960,
        float threshold = 0.15f,
        float boxThreshold = 0.3f,
        float unclipRatio = 1.6f)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        this.targetSize = targetSize;
        this.threshold = threshold;
        this.boxThreshold = boxThreshold;
        this.unclipRatio = unclipRatio;

        var sessionOptions = new SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        };

        session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <inheritdoc/>
    public DetectionResult Detect(byte[] imageData)
    {
        using var bitmap = ImageLoader.LoadWithOrientation(imageData);
        if (bitmap == null)
            throw new InvalidOperationException("Failed to decode image");
        
        return DetectInternal(bitmap);
    }

    /// <inheritdoc/>
    public IEnumerable<DetectionResult> DetectBatch(byte[][] imageBatch)
    {
        foreach (var imageData in imageBatch)
        {
            yield return Detect(imageData);
        }
    }

    /// <summary>
    /// Internal detection implementation
    /// </summary>
    private DetectionResult DetectInternal(SKBitmap originalBitmap)
    {
        var originalSize = (Width: originalBitmap.Width, Height: originalBitmap.Height);
        
        // Resize bitmap: scale longer side to targetSize
        var (resizedWidth, resizedHeight) = ImageResizer.CalculateResizeSize(
            originalBitmap.Width, 
            originalBitmap.Height, 
            targetSize);

        using var resizedBitmap = originalBitmap.Resize(
            new SKImageInfo(resizedWidth, resizedHeight), 
            new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
        
        if (resizedBitmap == null)
            throw new InvalidOperationException("Failed to resize bitmap");
        
        // Calculate padding to multiple of 32 (required for PP-OCR)
        var (paddedWidth, paddedHeight) = ImageResizer.CalculatePaddedSize(resizedWidth, resizedHeight);

        // Prepare tensor
        var inputTensor = ImageNormalizer.NormalizeToTensor(resizedBitmap, paddedWidth, paddedHeight);
        
        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(session.InputMetadata.First().Key, inputTensor)
        };

        using var results = session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();
        
        // Debug output analysis
        Console.WriteLine($"[DEBUG] Model Output Analysis:");
        Console.WriteLine($"  Output array length: {output.Length}");
        Console.WriteLine($"  Expected dimensions: {paddedWidth / 4}x{paddedHeight / 4} = {(paddedWidth / 4) * (paddedHeight / 4)}");
        Console.WriteLine($"  Min value: {output.Min():F6}");
        Console.WriteLine($"  Max value: {output.Max():F6}");
        Console.WriteLine($"  Mean value: {output.Average():F6}");

        // Analyze value distribution
        var aboveThreshold03 = output.Count(v => v > 0.3f);
        var aboveThreshold01 = output.Count(v => v > 0.1f);
        Console.WriteLine($"  Values > 0.3: {aboveThreshold03} ({100.0 * aboveThreshold03 / output.Length:F2}%)");
        Console.WriteLine($"  Values > 0.1: {aboveThreshold01} ({100.0 * aboveThreshold01 / output.Length:F2}%)");
        Console.WriteLine($"  Thresholds: detection={threshold}, box={boxThreshold}");
        
        // Extract bounding boxes
        var boxes = DetectionPostProcessor.ExtractBoxes(
            output, 
            paddedWidth, 
            paddedHeight, 
            resizedWidth, 
            resizedHeight, 
            originalSize,
            threshold,
            boxThreshold,
            unclipRatio);

        return new DetectionResult
        {
            Boxes = boxes,
            ProcessedImageSize = (paddedWidth, paddedHeight),
            OriginalImageSize = originalSize
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (disposed) return;
        
        session?.Dispose();
        disposed = true;
        GC.SuppressFinalize(this);
    }
}
