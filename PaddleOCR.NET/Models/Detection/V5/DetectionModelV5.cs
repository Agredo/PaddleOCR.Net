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
    /// <param name="threshold">Detection threshold (default: 0.3)</param>
    /// <param name="boxThreshold">Bounding box threshold (default: 0.5)</param>
    /// <param name="unclipRatio">Ratio for expanding boxes (default: 1.6)</param>
    public DetectionModelV5(
        string modelPath, 
        int targetSize = 960,
        float threshold = 0.3f,
        float boxThreshold = 0.5f,
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
        using var bitmap = SKBitmap.Decode(imageData);
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
        
        // Extract bounding boxes
        var boxes = DetectionPostProcessor.ExtractBoxes(
            output, 
            paddedWidth, 
            paddedHeight, 
            resizedWidth, 
            resizedHeight, 
            originalSize,
            threshold,
            boxThreshold);

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
