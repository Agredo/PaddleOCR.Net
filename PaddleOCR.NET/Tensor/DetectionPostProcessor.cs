using PaddleOCR.NET.Models.Detection;

namespace PaddleOCR.NET.Tensor;

/// <summary>
/// Post-processing utilities for detection model output
/// </summary>
public static class DetectionPostProcessor
{
    /// <summary>
    /// Extracts bounding boxes from detection model output
    /// </summary>
    /// <param name="output">Model output probability map</param>
    /// <param name="paddedWidth">Padded image width</param>
    /// <param name="paddedHeight">Padded image height</param>
    /// <param name="resizedWidth">Resized image width (before padding)</param>
    /// <param name="resizedHeight">Resized image height (before padding)</param>
    /// <param name="originalSize">Original image size</param>
    /// <param name="threshold">Detection threshold (default: 0.3)</param>
    /// <param name="boxThreshold">Box confidence threshold (default: 0.5)</param>
    /// <param name="unclipRatio">Ratio for expanding boxes (default: 1.6)</param>
    /// <param name="iouThreshold">IOU threshold for Non-Maximum Suppression (default: 0.3)</param>
    /// <returns>List of bounding boxes</returns>
    public static List<BoundingBox> ExtractBoxes(
        float[] output,
        int paddedWidth,
        int paddedHeight,
        int resizedWidth,
        int resizedHeight,
        (int Width, int Height) originalSize,
        float threshold = 0.3f,
        float boxThreshold = 0.5f,
        float unclipRatio = 1.6f,
        float iouThreshold = 0.3f)
    {
        var boxes = new List<BoundingBox>();

        // Output is a probability map - typically downsampled 4x
        var mapHeight = paddedHeight / 4;
        var mapWidth = paddedWidth / 4;

        Console.WriteLine($"[POST-PROCESS] Starting box extraction:");
        Console.WriteLine($"  Map dimensions: {mapWidth}x{mapHeight}");
        Console.WriteLine($"  Thresholds: detection={threshold}, box={boxThreshold}");
        Console.WriteLine($"  Unclip ratio: {unclipRatio}");

        // Verify output dimensions match expectations
        if (output.Length != mapWidth * mapHeight)
        {
            Console.WriteLine($"[WARNING] Output length mismatch!");
            Console.WriteLine($"  Expected: {mapWidth * mapHeight}, Got: {output.Length}");
            
            // Try to infer actual dimensions by testing common downsampling factors
            // The model processes the PADDED image and downsamples that
            bool dimensionsFound = false;
            
            // Try common downsampling factors (2x, 4x, 8x) from PADDED dimensions
            foreach (var downsample in new[] { 2, 4, 8 })
            {
                var testWidth = paddedWidth / downsample;
                var testHeight = paddedHeight / downsample;
                
                if (testWidth * testHeight == output.Length)
                {
                    mapWidth = testWidth;
                    mapHeight = testHeight;
                    Console.WriteLine($"[INFO] Adjusted to {downsample}x downsample from padded: {mapWidth}x{mapHeight}");
                    dimensionsFound = true;
                    break;
                }
            }
            
            // If common factors didn't work, try to find dimensions that match aspect ratio
            if (!dimensionsFound)
            {
                var targetAspectRatio = (float)paddedWidth / paddedHeight;
                var bestMatch = (width: mapWidth, height: mapHeight, diff: float.MaxValue);
                
                // Try different widths
                for (int w = 10; w <= paddedWidth; w++)
                {
                    if (output.Length % w == 0)
                    {
                        int h = output.Length / w;
                        var aspectRatio = (float)w / h;
                        var aspectDiff = Math.Abs(aspectRatio - targetAspectRatio);
                        
                        // Find best aspect ratio match
                        if (aspectDiff < bestMatch.diff)
                        {
                            bestMatch = (w, h, aspectDiff);
                        }
                    }
                }
                
                if (bestMatch.diff < 0.15f) // Allow some tolerance in aspect ratio
                {
                    mapWidth = bestMatch.width;
                    mapHeight = bestMatch.height;
                    var downsampleFactorX = (float)paddedWidth / mapWidth;
                    var downsampleFactorY = (float)paddedHeight / mapHeight;
                    Console.WriteLine($"[INFO] Adjusted to: {mapWidth}x{mapHeight} (downsample: {downsampleFactorX:F2}x{downsampleFactorY:F2})");
                    dimensionsFound = true;
                }
            }
            
            if (!dimensionsFound)
            {
                Console.WriteLine($"[ERROR] Could not determine valid map dimensions!");
                Console.WriteLine($"  Padded: {paddedWidth}x{paddedHeight}, Resized: {resizedWidth}x{resizedHeight}");
                Console.WriteLine($"  Output length: {output.Length}, sqrt: {Math.Sqrt(output.Length):F1}");
                return boxes;
            }
        }

        // Create binary map
        var binaryMap = new bool[mapHeight, mapWidth];
        var pixelsAboveThreshold = 0;
        
        for (int y = 0; y < mapHeight; y++)
        {
            for (int x = 0; x < mapWidth; x++)
            {
                var idx = y * mapWidth + x;
                if (idx < output.Length && output[idx] > threshold)
                {
                    binaryMap[y, x] = true;
                    pixelsAboveThreshold++;
                }
            }
        }

        Console.WriteLine($"  Binary map: {pixelsAboveThreshold} pixels above threshold ({100.0 * pixelsAboveThreshold / (mapWidth * mapHeight):F2}%)");

        if (pixelsAboveThreshold == 0)
        {
            Console.WriteLine("[WARNING] No pixels above threshold! Try lowering the threshold value.");
            return boxes;
        }

        // Extract contours
        var contours = ExtractContours(binaryMap);
        Console.WriteLine($"  Found {contours.Count} contours");

        // Calculate proper coordinate scaling:
        // The model output map dimensions were inferred above (mapWidth x mapHeight)
        // We need to scale from map space ? resized space ? original space
        
        // Step 1: Map ? Resized (direct scale based on actual dimensions)
        var mapToResizedScaleX = (float)resizedWidth / mapWidth;
        var mapToResizedScaleY = (float)resizedHeight / mapHeight;
        
        // Step 2: Resized ? Original
        var resizedToOriginalScaleX = (float)originalSize.Width / resizedWidth;
        var resizedToOriginalScaleY = (float)originalSize.Height / resizedHeight;

        Console.WriteLine($"  Dimensions: Original={originalSize.Width}x{originalSize.Height}, Resized={resizedWidth}x{resizedHeight}, Padded={paddedWidth}x{paddedHeight}, Map={mapWidth}x{mapHeight}");
        Console.WriteLine($"  Scaling: mapToResized=({mapToResizedScaleX:F3}, {mapToResizedScaleY:F3}), resizedToOriginal=({resizedToOriginalScaleX:F3}, {resizedToOriginalScaleY:F3})");

        int boxIndex = 0;
        foreach (var contour in contours)
        {
            if (contour.Count < 4)
            {
                Console.WriteLine($"  Contour {boxIndex}: {contour.Count} points - SKIPPED (too few points)");
                boxIndex++;
                continue;
            }

            // Get bounding rectangle in map space
            var minX = contour.Min(p => p.X);
            var maxX = contour.Max(p => p.X);
            var minY = contour.Min(p => p.Y);
            var maxY = contour.Max(p => p.Y);
            
            var mapWidth_box = maxX - minX + 1;
            var mapHeight_box = maxY - minY + 1;

            // Filter very small detections (likely noise)
            if (mapWidth_box < 2 || mapHeight_box < 2)
            {
                Console.WriteLine($"  Contour {boxIndex}: {mapWidth_box}x{mapHeight_box} - SKIPPED (too small)");
                boxIndex++;
                continue;
            }

            // Calculate confidence
            var confidence = CalculateConfidence(output, contour, mapWidth);
            
            // Scale from map space to resized space
            var resizedMinX = minX * mapToResizedScaleX;
            var resizedMaxX = maxX * mapToResizedScaleX;
            var resizedMinY = minY * mapToResizedScaleY;
            var resizedMaxY = maxY * mapToResizedScaleY;
            
            // Scale from resized space to original image space
            var originalMinX = resizedMinX * resizedToOriginalScaleX;
            var originalMaxX = resizedMaxX * resizedToOriginalScaleX;
            var originalMinY = resizedMinY * resizedToOriginalScaleY;
            var originalMaxY = resizedMaxY * resizedToOriginalScaleY;

            // Calculate center point and current dimensions
            var centerX = (originalMinX + originalMaxX) / 2f;
            var centerY = (originalMinY + originalMaxY) / 2f;
            var currentWidth = originalMaxX - originalMinX;
            var currentHeight = originalMaxY - originalMinY;

            // Expand box using unclipRatio
            var newWidth = currentWidth * unclipRatio;
            var newHeight = currentHeight * unclipRatio;

            // Calculate new corners around center point
            var expandedMinX = centerX - newWidth / 2f;
            var expandedMaxX = centerX + newWidth / 2f;
            var expandedMinY = centerY - newHeight / 2f;
            var expandedMaxY = centerY + newHeight / 2f;

            // Clamp to image bounds
            expandedMinX = Math.Max(0, expandedMinX);
            expandedMaxX = Math.Min(originalSize.Width, expandedMaxX);
            expandedMinY = Math.Max(0, expandedMinY);
            expandedMaxY = Math.Min(originalSize.Height, expandedMaxY);

            var points = new[]
            {
                new[] { expandedMinX, expandedMinY },
                new[] { expandedMaxX, expandedMinY },
                new[] { expandedMaxX, expandedMaxY },
                new[] { expandedMinX, expandedMaxY }
            };

            Console.WriteLine($"  Contour {boxIndex}: {contour.Count} pts, map=({minX},{minY})-({maxX},{maxY}), conf={confidence:F3}");
            Console.WriteLine($"    -> Resized: ({resizedMinX:F1},{resizedMinY:F1})-({resizedMaxX:F1},{resizedMaxY:F1})");
            Console.WriteLine($"    -> Original: ({originalMinX:F1},{originalMinY:F1})-({originalMaxX:F1},{originalMaxY:F1})");
            Console.WriteLine($"    -> Expanded ({unclipRatio}x): ({expandedMinX:F1},{expandedMinY:F1})-({expandedMaxX:F1},{expandedMaxY:F1})");

            if (confidence >= boxThreshold)
            {
                boxes.Add(new BoundingBox(points, confidence));
                Console.WriteLine($"    -> ADDED (confidence {confidence:F3} >= {boxThreshold})");
            }
            else
            {
                Console.WriteLine($"    -> SKIPPED (confidence {confidence:F3} < {boxThreshold})");
            }
            
            boxIndex++;
        }

        Console.WriteLine($"[POST-PROCESS] Extracted {boxes.Count} boxes from {contours.Count} contours");
        
        // Apply Non-Maximum Suppression to remove overlapping boxes
        if (boxes.Count > 0)
        {
            var boxesBeforeNMS = boxes.Count;
            boxes = ApplyNMS(boxes, iouThreshold);
            Console.WriteLine($"[POST-PROCESS] NMS: {boxesBeforeNMS} -> {boxes.Count} boxes (IOU threshold={iouThreshold})");
        }
        
        return boxes;
    }

    /// <summary>
    /// Extracts contours from a binary map using flood fill
    /// </summary>
    private static List<List<(int X, int Y)>> ExtractContours(bool[,] binaryMap)
    {
        var height = binaryMap.GetLength(0);
        var width = binaryMap.GetLength(1);
        var visited = new bool[height, width];
        var contours = new List<List<(int X, int Y)>>();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (binaryMap[y, x] && !visited[y, x])
                {
                    var contour = new List<(int X, int Y)>();
                    FloodFill(binaryMap, visited, x, y, contour);
                    
                    if (contour.Count >= 4)
                        contours.Add(contour);
                }
            }
        }

        return contours;
    }

    /// <summary>
    /// Performs flood fill to extract a contour
    /// </summary>
    private static void FloodFill(
        bool[,] map, 
        bool[,] visited, 
        int startX, 
        int startY, 
        List<(int X, int Y)> contour)
    {
        var height = map.GetLength(0);
        var width = map.GetLength(1);
        var stack = new Stack<(int X, int Y)>();
        stack.Push((startX, startY));

        while (stack.Count > 0)
        {
            var (x, y) = stack.Pop();
            
            if (x < 0 || x >= width || y < 0 || y >= height || visited[y, x] || !map[y, x])
                continue;

            visited[y, x] = true;
            contour.Add((x, y));

            // 8-neighborhood
            stack.Push((x + 1, y));
            stack.Push((x - 1, y));
            stack.Push((x, y + 1));
            stack.Push((x, y - 1));
            stack.Push((x + 1, y + 1));
            stack.Push((x + 1, y - 1));
            stack.Push((x - 1, y + 1));
            stack.Push((x - 1, y - 1));
        }
    }

    /// <summary>
    /// Calculates average confidence for a contour
    /// </summary>
    private static float CalculateConfidence(
        float[] output, 
        List<(int X, int Y)> contour, 
        int mapWidth)
    {
        if (contour.Count == 0) return 0f;

        var sum = 0f;
        foreach (var (x, y) in contour)
        {
            var idx = y * mapWidth + x;
            if (idx < output.Length)
                sum += output[idx];
        }

        return sum / contour.Count;
    }

    /// <summary>
    /// Applies Non-Maximum Suppression to remove overlapping bounding boxes
    /// </summary>
    /// <param name="boxes">List of bounding boxes to filter</param>
    /// <param name="iouThreshold">IOU threshold (boxes with IOU > threshold are suppressed)</param>
    /// <returns>Filtered list of bounding boxes</returns>
    private static List<BoundingBox> ApplyNMS(List<BoundingBox> boxes, float iouThreshold)
    {
        if (boxes.Count == 0)
            return boxes;

        // Sort boxes by confidence in descending order
        var sortedBoxes = boxes.OrderByDescending(b => b.Confidence).ToList();
        var keepBoxes = new List<BoundingBox>();
        var suppressed = new bool[sortedBoxes.Count];

        for (int i = 0; i < sortedBoxes.Count; i++)
        {
            if (suppressed[i])
                continue;

            keepBoxes.Add(sortedBoxes[i]);

            // Suppress boxes that overlap too much with this box
            for (int j = i + 1; j < sortedBoxes.Count; j++)
            {
                if (suppressed[j])
                    continue;

                var iou = CalculateIOU(sortedBoxes[i], sortedBoxes[j]);
                if (iou > iouThreshold)
                {
                    suppressed[j] = true;
                    Console.WriteLine($"  NMS: Suppressed box {j} (conf={sortedBoxes[j].Confidence:F3}, IOU={iou:F3} with box {i})");
                }
            }
        }

        return keepBoxes;
    }

    /// <summary>
    /// Calculates the Intersection over Union (IOU) between two bounding boxes
    /// </summary>
    /// <param name="box1">First bounding box</param>
    /// <param name="box2">Second bounding box</param>
    /// <returns>IOU value (0-1)</returns>
    private static float CalculateIOU(BoundingBox box1, BoundingBox box2)
    {
        var area1 = CalculateBoxArea(box1);
        var area2 = CalculateBoxArea(box2);
        
        if (area1 <= 0 || area2 <= 0)
            return 0f;

        var intersectionArea = CalculateIntersectionArea(box1, box2);
        var unionArea = area1 + area2 - intersectionArea;

        if (unionArea <= 0)
            return 0f;

        return intersectionArea / unionArea;
    }

    /// <summary>
    /// Calculates the area of a bounding box from its 4 corner points
    /// </summary>
    /// <param name="box">Bounding box</param>
    /// <returns>Area in square pixels</returns>
    private static float CalculateBoxArea(BoundingBox box)
    {
        // Use shoelace formula for polygon area
        // For a quadrilateral with points (x0,y0), (x1,y1), (x2,y2), (x3,y3):
        // Area = 0.5 * |x0(y1-y3) + x1(y2-y0) + x2(y3-y1) + x3(y0-y2)|
        
        var points = box.Points;
        if (points.Length != 4)
            return 0f;

        var area = 0.5f * Math.Abs(
            points[0].X * (points[1].Y - points[3].Y) +
            points[1].X * (points[2].Y - points[0].Y) +
            points[2].X * (points[3].Y - points[1].Y) +
            points[3].X * (points[0].Y - points[2].Y)
        );

        return area;
    }

    /// <summary>
    /// Calculates the intersection area between two bounding boxes
    /// </summary>
    /// <param name="box1">First bounding box</param>
    /// <param name="box2">Second bounding box</param>
    /// <returns>Intersection area in square pixels</returns>
    private static float CalculateIntersectionArea(BoundingBox box1, BoundingBox box2)
    {
        // For simplicity, approximate boxes as axis-aligned rectangles
        // Get min/max bounds for each box
        var box1MinX = box1.Points.Min(p => p.X);
        var box1MaxX = box1.Points.Max(p => p.X);
        var box1MinY = box1.Points.Min(p => p.Y);
        var box1MaxY = box1.Points.Max(p => p.Y);

        var box2MinX = box2.Points.Min(p => p.X);
        var box2MaxX = box2.Points.Max(p => p.X);
        var box2MinY = box2.Points.Min(p => p.Y);
        var box2MaxY = box2.Points.Max(p => p.Y);

        // Calculate intersection rectangle
        var intersectMinX = Math.Max(box1MinX, box2MinX);
        var intersectMaxX = Math.Min(box1MaxX, box2MaxX);
        var intersectMinY = Math.Max(box1MinY, box2MinY);
        var intersectMaxY = Math.Min(box1MaxY, box2MaxY);

        // Check if there's no intersection
        if (intersectMinX >= intersectMaxX || intersectMinY >= intersectMaxY)
            return 0f;

        var width = intersectMaxX - intersectMinX;
        var height = intersectMaxY - intersectMinY;

        return width * height;
    }
}
