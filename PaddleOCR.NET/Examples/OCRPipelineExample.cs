using PaddleOCR.NET.Models.Detection.V5;
using PaddleOCR.NET.Models.Recognition.V5;
using PaddleOCR.NET.Pipeline;

namespace PaddleOCR.NET.Examples;

/// <summary>
/// Complete OCR Pipeline Examples - Combining Detection and Recognition
/// </summary>
public static class OCRPipelineExample
{
    /// <summary>
    /// Example 1: Basic OCR Pipeline - Process a single image
    /// </summary>
    public static void BasicOCRPipeline()
    {
        // Build the complete OCR pipeline
        using var pipeline = new OCRPipelineBuilder()
            .WithDetection(builder => builder
                .WithModelPath("path/to/det.onnx")
                .WithThreshold(0.3f)
                .WithBoxThreshold(0.5f)
                .WithUnclipRatio(1.6f))
            .WithRecognition(builder => builder
                .WithModelPath("path/to/rec.onnx")
                .WithCharacterDict("path/to/dict.txt")
                .WithBatchSize(6))
            .Build();

        // Process an image
        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = pipeline.Process(imageBytes);

        // Display results
        Console.WriteLine($"Found {result.Count} text regions:\n");
        
        foreach (var region in result.TextRegions)
        {
            Console.WriteLine($"Region {region.Index + 1}:");
            Console.WriteLine($"  Text: \"{region.Text.Text}\"");
            Console.WriteLine($"  Confidence: {region.Text.Confidence:F3}");
            Console.WriteLine($"  Detection Confidence: {region.BoundingBox.Confidence:F3}");
            Console.WriteLine($"  Position: ({region.BoundingBox.Points[0].X:F0}, {region.BoundingBox.Points[0].Y:F0})");
            Console.WriteLine();
        }

        // Get full text
        Console.WriteLine("Full Text:");
        Console.WriteLine(result.GetFullText());
    }

    /// <summary>
    /// Example 2: Manual Pipeline - More control over each step
    /// </summary>
    public static void ManualOCRPipeline()
    {
        // Step 1: Create detection model
        using var detector = new DetectionModelV5Builder()
            .WithModelPath("path/to/det.onnx")
            .WithThreshold(0.3f)
            .WithBoxThreshold(0.5f)
            .Build();

        // Step 2: Create recognition model
        using var recognizer = new RecognitionModelV5Builder()
            .WithModelPath("path/to/rec.onnx")
            .WithCharacterDict("path/to/dict.txt")
            .WithBatchSize(6)
            .Build();

        // Step 3: Create pipeline
        using var pipeline = new OCRPipeline(detector, recognizer);

        // Step 4: Process image
        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = pipeline.Process(imageBytes);

        // Step 5: Process results
        foreach (var region in result.TextRegions)
        {
            Console.WriteLine($"{region.Text.Text} (confidence: {region.Text.Confidence:F3})");
        }
    }

    /// <summary>
    /// Example 3: Process multiple images (batch)
    /// </summary>
    public static void BatchOCRProcessing()
    {
        using var pipeline = new OCRPipelineBuilder()
            .WithDetection(builder => builder
                .WithModelPath("path/to/det.onnx")
                .WithThreshold(0.3f))
            .WithRecognition(builder => builder
                .WithModelPath("path/to/rec.onnx")
                .WithCharacterDict("path/to/dict.txt")
                .WithBatchSize(8)) // Larger batch for efficiency
            .Build();

        // Get all images from a directory
        var imageFiles = Directory.GetFiles("documents/", "*.jpg");
        var imageBatch = imageFiles.Select(File.ReadAllBytes).ToArray();

        // Process all images
        var results = pipeline.ProcessBatch(imageBatch);

        // Display results
        for (int i = 0; i < results.Length; i++)
        {
            Console.WriteLine($"\n=== Image: {Path.GetFileName(imageFiles[i])} ===");
            Console.WriteLine($"Found {results[i].Count} text regions");
            Console.WriteLine($"Text: {results[i].GetFullText(" ")}");
        }
    }

    /// <summary>
    /// Example 4: OCR with filtering and sorting
    /// </summary>
    public static void FilteredOCRProcessing()
    {
        using var pipeline = new OCRPipelineBuilder()
            .WithDetection(builder => builder
                .WithModelPath("path/to/det.onnx")
                .WithThreshold(0.3f))
            .WithRecognition(builder => builder
                .WithModelPath("path/to/rec.onnx")
                .WithCharacterDict("path/to/dict.txt"))
            .Build();

        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = pipeline.Process(imageBytes);

        // Filter by confidence threshold
        var highConfidenceRegions = result.TextRegions
            .Where(r => r.Text.Confidence > 0.8f)
            .ToList();

        Console.WriteLine($"High confidence regions: {highConfidenceRegions.Count}/{result.Count}");

        // Sort by vertical position (top to bottom)
        var sortedRegions = result.TextRegions
            .OrderBy(r => r.BoundingBox.Points[0].Y)
            .ThenBy(r => r.BoundingBox.Points[0].X)
            .ToList();

        Console.WriteLine("\nText in reading order:");
        foreach (var region in sortedRegions)
        {
            Console.WriteLine($"  {region.Text.Text}");
        }
    }

    /// <summary>
    /// Example 5: OCR with detailed analysis
    /// </summary>
    public static void DetailedOCRAnalysis()
    {
        using var pipeline = new OCRPipelineBuilder()
            .WithDetection(builder => builder
                .WithModelPath("path/to/det.onnx")
                .WithThreshold(0.3f)
                .WithBoxThreshold(0.5f))
            .WithRecognition(builder => builder
                .WithModelPath("path/to/rec.onnx")
                .WithCharacterDict("path/to/dict.txt"))
            .Build();

        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = pipeline.Process(imageBytes);

        Console.WriteLine($"=== OCR Analysis ===");
        Console.WriteLine($"Image Size: {result.DetectionResult.OriginalImageSize.Width}x{result.DetectionResult.OriginalImageSize.Height}");
        Console.WriteLine($"Total Regions: {result.Count}");
        Console.WriteLine($"Average Detection Confidence: {result.TextRegions.Average(r => r.BoundingBox.Confidence):F3}");
        Console.WriteLine($"Average Recognition Confidence: {result.TextRegions.Average(r => r.Text.Confidence):F3}");
        Console.WriteLine($"Total Characters: {result.TextRegions.Sum(r => r.Text.Text.Length)}");
        Console.WriteLine();

        // Show regions with low confidence
        var lowConfidenceRegions = result.TextRegions
            .Where(r => r.Text.Confidence < 0.7f)
            .ToList();

        if (lowConfidenceRegions.Any())
        {
            Console.WriteLine($"?? Low confidence regions ({lowConfidenceRegions.Count}):");
            foreach (var region in lowConfidenceRegions)
            {
                Console.WriteLine($"  \"{region.Text.Text}\" (confidence: {region.Text.Confidence:F3})");
            }
        }

        // Show per-character confidence for the first region
        if (result.Count > 0)
        {
            var firstRegion = result.TextRegions[0];
            Console.WriteLine($"\nFirst region character analysis:");
            Console.WriteLine($"Text: \"{firstRegion.Text.Text}\"");
            
            if (firstRegion.Text.CharConfidences != null)
            {
                for (int i = 0; i < firstRegion.Text.Text.Length; i++)
                {
                    Console.WriteLine($"  '{firstRegion.Text.Text[i]}': {firstRegion.Text.CharConfidences[i]:F3}");
                }
            }
        }
    }

    /// <summary>
    /// Example 6: OCR with box merging for better text line detection
    /// </summary>
    public static void OCRWithBoxMerging()
    {
        using var pipeline = new OCRPipelineBuilder()
            .WithDetection(builder => builder
                .WithModelPath("path/to/det.onnx")
                .WithThreshold(0.3f)
                .WithBoxThreshold(0.5f)
                .WithBoxMerging(true)                    // Enable box merging
                .WithMergeDistanceThreshold(0.5f)        // Merge nearby boxes
                .WithMergeOverlapThreshold(0.1f))        // Merge overlapping boxes
            .WithRecognition(builder => builder
                .WithModelPath("path/to/rec.onnx")
                .WithCharacterDict("path/to/dict.txt"))
            .Build();

        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = pipeline.Process(imageBytes);

        Console.WriteLine($"Found {result.Count} merged text lines:");
        foreach (var region in result.TextRegions)
        {
            Console.WriteLine($"  {region.Text.Text}");
        }
    }

    /// <summary>
    /// Example 7: Export OCR results to different formats
    /// </summary>
    public static void ExportOCRResults()
    {
        using var pipeline = new OCRPipelineBuilder()
            .WithDetection(builder => builder
                .WithModelPath("path/to/det.onnx"))
            .WithRecognition(builder => builder
                .WithModelPath("path/to/rec.onnx")
                .WithCharacterDict("path/to/dict.txt"))
            .Build();

        var imageBytes = File.ReadAllBytes("document.jpg");
        var result = pipeline.Process(imageBytes);

        // Export as plain text
        File.WriteAllText("output.txt", result.GetFullText());

        // Export as JSON
        var jsonData = System.Text.Json.JsonSerializer.Serialize(new
        {
            ImageSize = new
            {
                Width = result.DetectionResult.OriginalImageSize.Width,
                Height = result.DetectionResult.OriginalImageSize.Height
            },
            TextRegions = result.TextRegions.Select(r => new
            {
                Text = r.Text.Text,
                Confidence = r.Text.Confidence,
                BoundingBox = new
                {
                    Points = r.BoundingBox.Points.Select(p => new { X = p.X, Y = p.Y }).ToArray(),
                    Confidence = r.BoundingBox.Confidence
                }
            }).ToArray()
        }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });

        File.WriteAllText("output.json", jsonData);

        // Export as CSV
        var csvLines = new List<string> { "Index,Text,Confidence,X,Y" };
        csvLines.AddRange(result.TextRegions.Select(r =>
            $"{r.Index},\"{r.Text.Text.Replace("\"", "\"\"")}\",{r.Text.Confidence:F3},{r.BoundingBox.Points[0].X:F0},{r.BoundingBox.Points[0].Y:F0}"));

        File.WriteAllLines("output.csv", csvLines);

        Console.WriteLine("Results exported to:");
        Console.WriteLine("  - output.txt");
        Console.WriteLine("  - output.json");
        Console.WriteLine("  - output.csv");
    }
}
