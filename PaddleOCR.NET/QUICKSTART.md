# Complete OCR Pipeline - Quick Reference

## ?? Quick Start (30 seconds)

```csharp
using PaddleOCR.NET.Pipeline;

using var pipeline = new OCRPipelineBuilder()
    .WithDetection(b => b.WithModelPath("det.onnx").WithThreshold(0.3f))
    .WithRecognition(b => b.WithModelPath("rec.onnx").WithCharacterDict("dict.txt"))
    .Build();

var result = pipeline.Process(File.ReadAllBytes("image.jpg"));
Console.WriteLine(result.GetFullText());
```

## ?? Complete Example

```csharp
using PaddleOCR.NET.Models.Detection.V5;
using PaddleOCR.NET.Models.Recognition.V5;
using PaddleOCR.NET.Pipeline;
using PaddleOCR.NET.ImageProcessing;

// Method 1: Using Pipeline Builder (Recommended)
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(builder => builder
        .WithModelPath(@"C:\Models\det.onnx")
        .WithThreshold(0.3f)
        .WithBoxThreshold(0.5f)
        .WithUnclipRatio(1.6f)
        .WithBoxMerging(true)                    // Merge nearby boxes
        .WithMergeDistanceThreshold(0.5f))
    .WithRecognition(builder => builder
        .WithModelPath(@"C:\Models\rec.onnx")
        .WithCharacterDict(@"C:\Models\dict.txt")
        .WithBatchSize(6))
    .Build();

// Process image
var imageBytes = File.ReadAllBytes(@"C:\Images\document.jpg");
var result = pipeline.Process(imageBytes);

// Display results
Console.WriteLine($"Found {result.Count} text regions:");
foreach (var region in result.TextRegions)
{
    Console.WriteLine($"- \"{region.Text.Text}\" (confidence: {region.Text.Confidence:F3})");
}

// Get full text
Console.WriteLine("\nFull text:");
Console.WriteLine(result.GetFullText());

// Export annotated image
var outputPath = ImageExporter.ExportWithBoxes(
    @"C:\Images\document.jpg",
    result.DetectionResult,
    outputSuffix: "_ocr"
);
Console.WriteLine($"\nExported to: {outputPath}");
```

## ?? Common Scenarios

### 1. Document Scanning (Text Lines)

```csharp
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(b => b
        .WithModelPath("det.onnx")
        .WithThreshold(0.3f)
        .WithBoxMerging(true)           // Enable line merging
        .WithMergeDistanceThreshold(0.5f))
    .WithRecognition(b => b
        .WithModelPath("rec.onnx")
        .WithCharacterDict("dict.txt"))
    .Build();

var result = pipeline.Process(imageBytes);
var lines = result.TextRegions
    .OrderBy(r => r.BoundingBox.Points[0].Y)  // Top to bottom
    .Select(r => r.Text.Text);

File.WriteAllLines("output.txt", lines);
```

### 2. Scene Text (Individual Words)

```csharp
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(b => b
        .WithModelPath("det.onnx")
        .WithThreshold(0.5f)            // Higher threshold
        .WithBoxMerging(false))         // Keep individual words
    .WithRecognition(b => b
        .WithModelPath("rec.onnx")
        .WithCharacterDict("dict.txt"))
    .Build();

var result = pipeline.Process(imageBytes);
foreach (var word in result.TextRegions)
{
    Console.WriteLine($"{word.Text.Text} @ ({word.BoundingBox.Points[0].X:F0}, {word.BoundingBox.Points[0].Y:F0})");
}
```

### 3. Batch Processing (Multiple Images)

```csharp
using var pipeline = new OCRPipelineBuilder()
    .WithDetection(b => b.WithModelPath("det.onnx"))
    .WithRecognition(b => b
        .WithModelPath("rec.onnx")
        .WithCharacterDict("dict.txt")
        .WithBatchSize(8))              // Larger batch
    .Build();

var images = Directory.GetFiles(@"C:\Images", "*.jpg")
    .Select(File.ReadAllBytes)
    .ToArray();

var results = pipeline.ProcessBatch(images);

for (int i = 0; i < results.Length; i++)
{
    Console.WriteLine($"\nImage {i + 1}: {results[i].Count} regions");
    Console.WriteLine(results[i].GetFullText());
}
```

### 4. Filter by Confidence

```csharp
var result = pipeline.Process(imageBytes);

var highQuality = result.TextRegions
    .Where(r => r.Text.Confidence > 0.8f)           // Good recognition
    .Where(r => r.BoundingBox.Confidence > 0.7f)    // Good detection
    .ToList();

Console.WriteLine($"High quality: {highQuality.Count}/{result.Count}");
```

### 5. Export Multiple Formats

```csharp
var result = pipeline.Process(imageBytes);

// Plain text
File.WriteAllText("output.txt", result.GetFullText());

// JSON
var json = System.Text.Json.JsonSerializer.Serialize(
    result.TextRegions.Select(r => new { r.Text.Text, r.Text.Confidence }),
    new System.Text.Json.JsonSerializerOptions { WriteIndented = true }
);
File.WriteAllText("output.json", json);

// CSV
var csv = new[] { "Text,Confidence,X,Y" }
    .Concat(result.TextRegions.Select(r =>
        $"\"{r.Text.Text}\",{r.Text.Confidence},{r.BoundingBox.Points[0].X},{r.BoundingBox.Points[0].Y}"));
File.WriteAllLines("output.csv", csv);
```

## ?? API Reference

### OCRPipeline Methods

```csharp
// Single image
OCRResult Process(byte[] imageData)

// Multiple images
OCRResult[] ProcessBatch(byte[][] imageBatch)
```

### OCRResult Properties

```csharp
result.Count                      // Number of text regions
result.TextRegions                // List of OCRTextRegion
result.DetectionResult            // Original DetectionResult
result.GetFullText("\n")          // All text concatenated
```

### OCRTextRegion Properties

```csharp
region.Index                      // Index in results
region.Text.Text                  // Recognized text string
region.Text.Confidence            // Recognition confidence (0-1)
region.Text.CharConfidences       // Per-character confidence
region.BoundingBox.Points         // Corner coordinates
region.BoundingBox.Confidence     // Detection confidence
```

## ?? Required Files

```
Your Project/
??? det.onnx              ? Detection model (required)
??? rec.onnx              ? Recognition model (required)
??? dict.txt              ? Character dictionary (required)
```

### Where to Get Models

1. **Official PaddleOCR Models**: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
2. **ONNX Converted Models**: https://github.com/RapidAI/RapidOCR

## ?? Configuration Guide

### Detection Settings

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `Threshold` | 0.15 | Detection sensitivity | Lower (0.3) for faint text |
| `BoxThreshold` | 0.3 | Box confidence | Lower (0.2) to detect more regions |
| `UnclipRatio` | 1.6 | Box expansion | Increase (2.0) if text is cut off |
| `BoxMerging` | false | Merge nearby boxes | Enable for line detection |
| `MergeDistanceThreshold` | 0.5 | Distance for merging | Higher = more aggressive merging |
| `TargetSize` | 960 | Image preprocessing size | Larger = slower but more accurate |

### Recognition Settings

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `BatchSize` | 6 | Images per batch | Increase for more throughput |

## ?? Result Sorting & Filtering

### Sort by Position (Reading Order)

```csharp
var ordered = result.TextRegions
    .OrderBy(r => r.BoundingBox.Points[0].Y)    // Top to bottom
    .ThenBy(r => r.BoundingBox.Points[0].X);    // Left to right
```

### Filter by Confidence

```csharp
var reliable = result.TextRegions
    .Where(r => r.Text.Confidence > 0.8f);
```

### Group by Vertical Position (Lines)

```csharp
var lines = result.TextRegions
    .GroupBy(r => (int)(r.BoundingBox.Points[0].Y / 50))  // Group by 50px
    .Select(g => string.Join(" ", g.OrderBy(r => r.BoundingBox.Points[0].X)
                                   .Select(r => r.Text.Text)));
```

## ?? Troubleshooting

| Problem | Solution |
|---------|----------|
| No text detected | Lower detection threshold to 0.2-0.3 |
| Wrong language recognized | Check character dictionary matches image language |
| Boxes too small | Increase `UnclipRatio` to 2.0 |
| Boxes too large | Decrease `UnclipRatio` to 1.2 |
| Multiple boxes per word | Enable `BoxMerging`, increase `MergeDistanceThreshold` |
| Slow processing | Decrease `TargetSize`, increase `BatchSize` |
| Out of memory | Decrease `BatchSize` |
| Low confidence | Improve image quality, check model/dict match |

## ?? Performance Tips

1. **Reuse Pipeline**: Create once, use for many images
2. **Batch Processing**: Process multiple images together
3. **Appropriate Thresholds**: Balance accuracy vs speed
4. **Image Size**: Don't make images larger than needed
5. **Memory Management**: Use `using` statements for disposal

## ?? More Resources

- **Detection Details**: `Models/Detection/README.md`
- **Recognition Details**: `Models/Recognition/README.md`
- **Pipeline Guide**: `Pipeline/README.md`
- **Examples**: `Examples/OCRPipelineExample.cs`
- **Console Test**: `PaddleOCR.NET.Console/Program.cs`

## ?? Full Example in Program.cs

Uncomment `TestCompleteOCRPipeline()` in `Program.cs` to run a complete test with:
- Detection
- Recognition
- Result display
- Statistics
- Image export
- Text export

```csharp
// TestCompleteOCRPipeline();
```
