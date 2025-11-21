using PaddleOCR.NET.Models;
using PaddleOCR.NET.Models.Detection.V5;

namespace PaddleOCR.NET.Examples;

/// <summary>
/// Example usage of the DetectionModelV5
/// </summary>
public static class DetectionExample
{
    /// <summary>
    /// Basic detection example
    /// </summary>
    public static void BasicDetection()
    {
        // Create detection model
        using var detector = new DetectionModelV5("path/to/detection/v5/det.onnx");

        // Load image
        var imageBytes = File.ReadAllBytes("document.jpg");

        // Detect text regions
        var result = detector.Detect(imageBytes);

        // Process results
        Console.WriteLine($"Original Size: {result.OriginalImageSize.Width}x{result.OriginalImageSize.Height}");
        Console.WriteLine($"Processed Size: {result.ProcessedImageSize.Width}x{result.ProcessedImageSize.Height}");
        Console.WriteLine($"Found {result.Boxes.Count} text regions\n");

        foreach (var box in result.Boxes)
        {
            Console.WriteLine($"Confidence: {box.Confidence:F2}");
            for (int i = 0; i < box.Points.Length; i++)
            {
                Console.WriteLine($"  Point {i + 1}: ({box.Points[i].X:F1}, {box.Points[i].Y:F1})");
            }
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Detection with custom configuration using builder
    /// </summary>
    public static void DetectionWithBuilder()
    {
        // Create detector with custom configuration
        using var detector = new DetectionModelV5Builder()
            .WithModelPath("path/to/detection/v5/det.onnx")
            .WithTargetSize(960)         // Longer side target size
            .WithThreshold(0.3f)         // Detection threshold
            .WithBoxThreshold(0.5f)      // Box confidence threshold
            .WithUnclipRatio(1.6f)       // Box expansion ratio
            .Build();

        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = detector.Detect(imageBytes);

        Console.WriteLine($"Detected {result.Boxes.Count} text regions");
    }

    /// <summary>
    /// Batch detection example
    /// </summary>
    public static void BatchDetection()
    {
        using var detector = new DetectionModelV5("path/to/detection/v5/det.onnx");

        var imageBatch = new[]
        {
            File.ReadAllBytes("image1.jpg"),
            File.ReadAllBytes("image2.jpg"),
            File.ReadAllBytes("image3.jpg")
        };

        var results = detector.DetectBatch(imageBatch);

        int imageIndex = 1;
        foreach (var result in results)
        {
            Console.WriteLine($"Image {imageIndex}: Found {result.Boxes.Count} text regions");
            imageIndex++;
        }
    }

    /// <summary>
    /// Filter results by confidence
    /// </summary>
    public static void FilterByConfidence()
    {
        using var detector = new DetectionModelV5Builder()
            .WithModelPath("path/to/detection/v5/det.onnx")
            .WithBoxThreshold(0.7f) // Higher threshold for more confident detections
            .Build();

        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = detector.Detect(imageBytes);

        // All boxes already filtered by boxThreshold (0.7)
        var highConfidenceBoxes = result.Boxes
            .Where(box => box.Confidence >= 0.8f)
            .OrderByDescending(box => box.Confidence);

        foreach (var box in highConfidenceBoxes)
        {
            Console.WriteLine($"High confidence box: {box.Confidence:F2}");
        }
    }

    /// <summary>
    /// Process different image sizes
    /// </summary>
    public static void DifferentImageSizes()
    {
        // For smaller images
        using var detectorSmall = new DetectionModelV5Builder()
            .WithModelPath("path/to/detection/v5/det.onnx")
            .WithTargetSize(640)
            .Build();

        // For larger images
        using var detectorLarge = new DetectionModelV5Builder()
            .WithModelPath("path/to/detection/v5/det.onnx")
            .WithTargetSize(1280)
            .Build();

        var smallImage = File.ReadAllBytes("small_document.jpg");
        var largeImage = File.ReadAllBytes("large_document.jpg");

        var resultSmall = detectorSmall.Detect(smallImage);
        var resultLarge = detectorLarge.Detect(largeImage);

        Console.WriteLine($"Small image: {resultSmall.Boxes.Count} boxes");
        Console.WriteLine($"Large image: {resultLarge.Boxes.Count} boxes");
    }
}
