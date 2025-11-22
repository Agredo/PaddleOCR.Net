# PP-OCR Recognition Model

This module implements text recognition for PaddleOCR.NET using the PP-OCRv5 recognition model.

## Features

- ? Single image text recognition
- ? Batch processing for multiple images
- ? CTC decoding with confidence scores
- ? Per-character confidence tracking
- ? Configurable batch size for optimal performance
- ? Fluent builder pattern for easy configuration

## Quick Start

```csharp
using PaddleOCR.NET.Models.Recognition.V5;

// Create recognition model
using var recognizer = new RecognitionModelV5Builder()
    .WithModelPath("path/to/rec.onnx")
    .WithCharacterDict("path/to/ppocr_keys_v1.txt")
    .WithBatchSize(6)
    .Build();

// Recognize text in single image
var imageBytes = File.ReadAllBytes("text_image.jpg");
var result = recognizer.Recognize(imageBytes);

Console.WriteLine($"Text: {result.Text}");
Console.WriteLine($"Confidence: {result.Confidence:F3}");
```

## Architecture

### File Structure

```
Models/Recognition/
??? IRecognitionModel.cs          - Interface for recognition models
??? RecognizedText.cs             - Single recognition result
??? RecognitionResult.cs          - Batch recognition results
??? V5/
    ??? RecognitionModelV5.cs     - Main model implementation
    ??? RecognitionModelV5Builder.cs - Fluent builder
    ??? RecognitionPreProcessor.cs   - Image preprocessing

Tensor/
??? RecognitionPostProcessor.cs   - CTC decoding logic
```

### Components

#### 1. RecognizedText
Contains the result of text recognition for a single image:
- `Text`: The recognized text string
- `Confidence`: Overall confidence score (0-1)
- `CharConfidences`: Per-character confidence scores (optional)

#### 2. RecognitionResult
Contains results from batch processing:
- `Texts`: List of RecognizedText objects
- `Count`: Number of recognized texts

#### 3. IRecognitionModel
Interface defining the recognition contract:
- `Recognize(byte[] imageData)`: Single image recognition
- `RecognizeBatch(byte[][] imageBatch)`: Batch recognition from bytes
- `RecognizeBatch(SKBitmap[] bitmaps)`: Batch recognition from bitmaps

## Preprocessing

The recognition preprocessing follows the PP-OCR standard:

### Image Resizing
- **Fixed height**: 48 pixels
- **Variable width**: Calculated based on aspect ratio
- **Max width**: 320 pixels
- **Padding**: Right-side padding to max width

### Normalization
Formula: `(pixel / 255.0 - 0.5) / 0.5`

This transforms pixel values from [0, 255] to [-1, 1] range.

### Input Tensor Shape
`[batch_size, 3, 48, 320]`

## Post-processing (CTC Decoding)

The recognition uses CTC (Connectionist Temporal Classification) decoding:

### Algorithm Steps
1. **Get predictions**: Find character with highest probability at each time step
2. **Remove duplicates**: Collapse consecutive identical characters
3. **Remove blanks**: Filter out blank tokens (index 0)
4. **Calculate confidence**: Average of remaining character probabilities

### Character Dictionary
- **Format**: UTF-8 text file with one character per line
- **Index 0**: Reserved for blank token (automatically added)
- **Index 1+**: Characters from the dictionary file

Example dictionary structure:
```
0
1
2
...
a
b
c
...
?
?
?
```

## Usage Examples

### Single Image Recognition

```csharp
using var recognizer = new RecognitionModelV5Builder()
    .WithModelPath("rec.onnx")
    .WithCharacterDict("ppocr_keys_v1.txt")
    .Build();

var result = recognizer.Recognize(imageBytes);
Console.WriteLine($"Text: {result.Text}");
Console.WriteLine($"Confidence: {result.Confidence:F3}");

// Print per-character confidences
if (result.CharConfidences != null)
{
    for (int i = 0; i < result.Text.Length; i++)
    {
        Console.WriteLine($"'{result.Text[i]}': {result.CharConfidences[i]:F3}");
    }
}
```

### Batch Processing

```csharp
using var recognizer = new RecognitionModelV5Builder()
    .WithModelPath("rec.onnx")
    .WithCharacterDict("ppocr_keys_v1.txt")
    .WithBatchSize(8)  // Process 8 images at a time
    .Build();

var imagePaths = Directory.GetFiles("text_images/", "*.jpg");
var imageBatch = imagePaths.Select(File.ReadAllBytes).ToArray();

var results = recognizer.RecognizeBatch(imageBatch);

foreach (var text in results.Texts)
{
    Console.WriteLine($"\"{text.Text}\" (confidence: {text.Confidence:F3})");
}
```

### Complete OCR Pipeline (Detection + Recognition)

```csharp
// 1. Detect text regions
using var detector = new DetectionModelV5Builder()
    .WithModelPath("det.onnx")
    .Build();

using var recognizer = new RecognitionModelV5Builder()
    .WithModelPath("rec.onnx")
    .WithCharacterDict("ppocr_keys_v1.txt")
    .Build();

var imageBytes = File.ReadAllBytes("document.jpg");
var detectionResult = detector.Detect(imageBytes);

// 2. For each detected box, crop and recognize
// (Image cropping functionality would be needed here)
foreach (var box in detectionResult.Boxes)
{
    // TODO: Crop image region using bounding box
    // var croppedBytes = CropImage(imageBytes, box);
    // var text = recognizer.Recognize(croppedBytes);
    // Console.WriteLine($"Detected text: {text.Text}");
}
```

## Configuration

### Builder Options

| Method | Description | Default |
|--------|-------------|---------|
| `WithModelPath(string)` | Path to rec.onnx file | Required |
| `WithCharacterDict(string)` | Path to character dictionary | Required |
| `WithBatchSize(int)` | Batch size for processing | 6 |

### Batch Size Recommendations

- **Small images (< 100 chars)**: 8-16
- **Medium images (100-200 chars)**: 4-8
- **Large images (> 200 chars)**: 2-4

Higher batch sizes improve throughput but require more memory.

## Model Specifications

### Input
- **Tensor shape**: [batch_size, 3, 48, 320]
- **Data type**: float32
- **Value range**: [-1, 1]

### Output
- **Tensor shape**: [batch_size, sequence_length, num_classes]
- **Data type**: float32
- **Values**: Softmax probabilities

### Typical Values
- **Sequence length**: ~80 (varies by model)
- **Number of classes**: 6623+ (depends on dictionary)

## Performance Tips

1. **Use batch processing** when recognizing multiple images
2. **Adjust batch size** based on available memory and image complexity
3. **Reuse model instances** - avoid creating new models for each recognition
4. **Pre-load images** into memory for faster batch processing

## Error Handling

The recognition model throws exceptions for:
- Missing model file (`FileNotFoundException`)
- Missing character dictionary (`FileNotFoundException`)
- Invalid batch size (< 1) (`ArgumentException`)
- Failed image decoding (`InvalidOperationException`)
- Empty batch array (`ArgumentException`)

## Testing Recommendations

Test with:
- ? Single characters
- ? Short words (2-5 characters)
- ? Long text (10+ characters)
- ? Various aspect ratios (narrow to wide)
- ? Different languages (if supported by dictionary)
- ? Batch processing with mixed image sizes

## References

- Based on RapidOCR implementation
- PP-OCR paper: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- CTC decoding: [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
