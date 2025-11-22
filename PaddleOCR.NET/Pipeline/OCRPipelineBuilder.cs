using PaddleOCR.NET.Models.Detection.V5;
using PaddleOCR.NET.Models.Recognition.V5;

namespace PaddleOCR.NET.Pipeline;

/// <summary>
/// Fluent builder for OCRPipeline
/// </summary>
public class OCRPipelineBuilder
{
    private DetectionModelV5Builder? detectionBuilder;
    private RecognitionModelV5Builder? recognitionBuilder;

    /// <summary>
    /// Creates a new pipeline builder
    /// </summary>
    public OCRPipelineBuilder()
    {
        detectionBuilder = new DetectionModelV5Builder();
        recognitionBuilder = new RecognitionModelV5Builder();
    }

    /// <summary>
    /// Configures the detection model
    /// </summary>
    /// <param name="configure">Action to configure the detection builder</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public OCRPipelineBuilder WithDetection(Action<DetectionModelV5Builder> configure)
    {
        if (configure == null)
            throw new ArgumentNullException(nameof(configure));
        
        detectionBuilder = new DetectionModelV5Builder();
        configure(detectionBuilder);
        return this;
    }

    /// <summary>
    /// Configures the recognition model
    /// </summary>
    /// <param name="configure">Action to configure the recognition builder</param>
    /// <returns>Builder instance for fluent chaining</returns>
    public OCRPipelineBuilder WithRecognition(Action<RecognitionModelV5Builder> configure)
    {
        if (configure == null)
            throw new ArgumentNullException(nameof(configure));
        
        recognitionBuilder = new RecognitionModelV5Builder();
        configure(recognitionBuilder);
        return this;
    }

    /// <summary>
    /// Builds the OCR pipeline
    /// </summary>
    /// <returns>Configured OCR pipeline instance</returns>
    public OCRPipeline Build()
    {
        if (detectionBuilder == null)
            throw new InvalidOperationException("Detection model must be configured");
        
        if (recognitionBuilder == null)
            throw new InvalidOperationException("Recognition model must be configured");

        var detectionModel = detectionBuilder.Build();
        var recognitionModel = recognitionBuilder.Build();

        return new OCRPipeline(detectionModel, recognitionModel);
    }
}
