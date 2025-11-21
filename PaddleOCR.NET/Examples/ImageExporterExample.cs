using PaddleOCR.NET.ImageProcessing;
using PaddleOCR.NET.Models.Detection.V5;
using SkiaSharp;

namespace PaddleOCR.NET.Examples;

/// <summary>
/// Examples demonstrating how to use the ImageExporter
/// </summary>
public static class ImageExporterExample
{
    /// <summary>
    /// Basic example: Export image with detected boxes
    /// </summary>
    public static void BasicExport()
    {
        var imagePath = @"C:\path\to\image.png";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        // Export with default settings
        var outputPath = ImageExporter.ExportWithBoxes(imagePath, result);
        Console.WriteLine($"Saved to: {outputPath}");
    }
    
    /// <summary>
    /// Custom styling: Thick lines and custom suffix
    /// </summary>
    public static void CustomStyling()
    {
        var imagePath = @"C:\path\to\image.png";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        // Export with custom styling
        var outputPath = ImageExporter.ExportWithBoxes(
            originalImagePath: imagePath,
            detectionResult: result,
            outputSuffix: "_boxes",
            strokeWidth: 5f,
            showConfidence: true,
            quality: 90
        );
        
        Console.WriteLine($"Saved to: {outputPath}");
    }
    
    /// <summary>
    /// Export without confidence labels
    /// </summary>
    public static void WithoutConfidence()
    {
        var imagePath = @"C:\path\to\image.png";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        // Export without confidence text
        var outputPath = ImageExporter.ExportWithBoxes(
            originalImagePath: imagePath,
            detectionResult: result,
            showConfidence: false
        );
        
        Console.WriteLine($"Saved to: {outputPath}");
    }
    
    /// <summary>
    /// Export from byte array to custom location
    /// </summary>
    public static void ExportFromByteArray()
    {
        var imagePath = @"C:\path\to\image.png";
        var outputPath = @"C:\output\result.png";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        // Export from byte array to specific path
        ImageExporter.ExportWithBoxes(
            imageData: imageBytes,
            outputPath: outputPath,
            detectionResult: result,
            format: SKEncodedImageFormat.Png
        );
        
        Console.WriteLine($"Saved to: {outputPath}");
    }
    
    /// <summary>
    /// Batch processing with exports
    /// </summary>
    public static void BatchProcessing()
    {
        var imagePaths = Directory.GetFiles(@"C:\images", "*.png");
        
        using var detector = new DetectionModelV5Builder()
            .WithModelPath(@"C:\path\to\det.onnx")
            .WithThreshold(0.3f)
            .WithBoxThreshold(0.5f)
            .Build();
        
        foreach (var imagePath in imagePaths)
        {
            try
            {
                var imageBytes = File.ReadAllBytes(imagePath);
                var result = detector.Detect(imageBytes);
                
                if (result.Boxes.Count > 0)
                {
                    var outputPath = ImageExporter.ExportWithBoxes(imagePath, result);
                    Console.WriteLine($"? Processed: {Path.GetFileName(imagePath)} ({result.Boxes.Count} boxes)");
                }
                else
                {
                    Console.WriteLine($"? No text detected: {Path.GetFileName(imagePath)}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"? Error processing {Path.GetFileName(imagePath)}: {ex.Message}");
            }
        }
    }
    
    /// <summary>
    /// Export different formats
    /// </summary>
    public static void ExportDifferentFormats()
    {
        var imagePath = @"C:\path\to\image.png";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        // Export as PNG
        ImageExporter.ExportWithBoxes(
            imageData: imageBytes,
            outputPath: @"C:\output\result.png",
            detectionResult: result,
            format: SKEncodedImageFormat.Png
        );
        
        // Export as JPEG with high quality
        ImageExporter.ExportWithBoxes(
            imageData: imageBytes,
            outputPath: @"C:\output\result.jpg",
            detectionResult: result,
            format: SKEncodedImageFormat.Jpeg,
            quality: 95
        );
        
        // Export as WebP
        ImageExporter.ExportWithBoxes(
            imageData: imageBytes,
            outputPath: @"C:\output\result.webp",
            detectionResult: result,
            format: SKEncodedImageFormat.Webp,
            quality: 90
        );
    }
}
