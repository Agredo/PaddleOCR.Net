using PaddleOCR.NET.Models.Detection;
using Clipper2Lib;
using System.Drawing;

namespace PaddleOCR.NET.Tensor;

/// <summary>
/// Post-processing utilities for detection model output
/// </summary>
public static class DetectionPostProcessor
{
    /// <summary>
    /// Represents a minimum area rotated rectangle
    /// </summary>
    public class MinimumAreaRectangle
    {
        /// <summary>
        /// Center point of the rectangle
        /// </summary>
        public PointF Center { get; init; }

        /// <summary>
        /// Size of the rectangle (width, height)
        /// </summary>
        public SizeF Size { get; init; }

        /// <summary>
        /// Rotation angle in degrees (0-180)
        /// </summary>
        public float Angle { get; init; }

        /// <summary>
        /// Gets the four corner points of the rectangle in clockwise order starting from top-left
        /// </summary>
        public PointF[] GetCornerPoints()
        {
            var halfWidth = Size.Width / 2f;
            var halfHeight = Size.Height / 2f;

            // Create corners in local space (centered at origin)
            var corners = new[]
            {
                new PointF(-halfWidth, -halfHeight), // Top-left
                new PointF(halfWidth, -halfHeight),  // Top-right
                new PointF(halfWidth, halfHeight),   // Bottom-right
                new PointF(-halfWidth, halfHeight)   // Bottom-left
            };

            // Rotate and translate to center position
            var angleRad = Angle * Math.PI / 180.0;
            var cos = (float)Math.Cos(angleRad);
            var sin = (float)Math.Sin(angleRad);

            for (int i = 0; i < corners.Length; i++)
            {
                var x = corners[i].X * cos - corners[i].Y * sin + Center.X;
                var y = corners[i].X * sin + corners[i].Y * cos + Center.Y;
                corners[i] = new PointF(x, y);
            }

            return corners;
        }

        /// <summary>
        /// Finds the minimum area rectangle that encloses a contour
        /// </summary>
        public static MinimumAreaRectangle FromContour(List<(int X, int Y)> contour)
        {
            if (contour.Count < 3)
            {
                // Degenerate case - return simple bounds
                var minX = contour.Min(p => p.X);
                var maxX = contour.Max(p => p.X);
                var minY = contour.Min(p => p.Y);
                var maxY = contour.Max(p => p.Y);
                
                return new MinimumAreaRectangle
                {
                    Center = new PointF((minX + maxX) / 2f, (minY + maxY) / 2f),
                    Size = new SizeF(maxX - minX, maxY - minY),
                    Angle = 0f
                };
            }

            // Convert to PointF for calculations
            var points = contour.Select(p => new PointF(p.X, p.Y)).ToList();

            // Calculate centroid
            var centroidX = points.Average(p => p.X);
            var centroidY = points.Average(p => p.Y);
            var centroid = new PointF(centroidX, centroidY);

            // Try different rotation angles to find minimum area
            var minArea = float.MaxValue;
            var bestAngle = 0f;
            var bestBounds = RectangleF.Empty;

            // Test angles from 0° to 180° in 5° increments
            for (int angleDeg = 0; angleDeg < 180; angleDeg += 5)
            {
                var angleRad = angleDeg * Math.PI / 180.0;
                var cos = Math.Cos(angleRad);
                var sin = Math.Sin(angleRad);

                // Rotate all points around centroid
                var rotatedPoints = new List<PointF>();
                foreach (var point in points)
                {
                    var dx = point.X - centroid.X;
                    var dy = point.Y - centroidY;
                    var rotatedX = (float)(dx * cos - dy * sin);
                    var rotatedY = (float)(dx * sin + dy * cos);
                    rotatedPoints.Add(new PointF(rotatedX, rotatedY));
                }

                // Get axis-aligned bounds of rotated points
                var bounds = GetAxisAlignedBounds(rotatedPoints);
                var area = bounds.Width * bounds.Height;

                if (area < minArea)
                {
                    minArea = area;
                    bestAngle = angleDeg;
                    bestBounds = bounds;
                }
            }

            return new MinimumAreaRectangle
            {
                Center = centroid,
                Size = bestBounds.Size,
                Angle = bestAngle
            };
        }
    }

    /// <summary>
    /// Dilates a binary map using morphological dilation
    /// </summary>
    /// <param name="binaryMap">Input binary map</param>
    /// <param name="kernelSize">Size of the dilation kernel (default: 2)</param>
    /// <returns>Dilated binary map</returns>
    private static bool[,] DilateBinaryMap(bool[,] binaryMap, int kernelSize = 2)
    {
        var height = binaryMap.GetLength(0);
        var width = binaryMap.GetLength(1);
        var dilated = new bool[height, width];

        var halfKernel = kernelSize / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Check if any pixel in the kernel neighborhood is set
                var shouldSet = false;
                for (int ky = -halfKernel; ky < kernelSize - halfKernel; ky++)
                {
                    for (int kx = -halfKernel; kx < kernelSize - halfKernel; kx++)
                    {
                        var ny = y + ky;
                        var nx = x + kx;
                        
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width && binaryMap[ny, nx])
                        {
                            shouldSet = true;
                            break;
                        }
                    }
                    if (shouldSet) break;
                }

                dilated[y, x] = shouldSet;
            }
        }

        return dilated;
    }

    /// <summary>
    /// Calculates the area of a polygon using the Shoelace formula
    /// </summary>
    private static float CalculatePolygonArea(List<(float X, float Y)> polygon)
    {
        if (polygon.Count < 3)
            return 0f;

        var area = 0f;
        for (int i = 0; i < polygon.Count; i++)
        {
            var j = (i + 1) % polygon.Count;
            area += polygon[i].X * polygon[j].Y;
            area -= polygon[j].X * polygon[i].Y;
        }

        return Math.Abs(area) / 2f;
    }

    /// <summary>
    /// Calculates the perimeter of a polygon
    /// </summary>
    private static float CalculatePolygonPerimeter(List<(float X, float Y)> polygon)
    {
        if (polygon.Count < 2)
            return 0f;

        var perimeter = 0f;
        for (int i = 0; i < polygon.Count; i++)
        {
            var j = (i + 1) % polygon.Count;
            var dx = polygon[j].X - polygon[i].X;
            var dy = polygon[j].Y - polygon[i].Y;
            perimeter += (float)Math.Sqrt(dx * dx + dy * dy);
        }

        return perimeter;
    }

    /// <summary>
    /// Expands a polygon using Clipper2 offset algorithm
    /// </summary>
    private static List<(float X, float Y)> ExpandPolygonWithClipper(
        List<(float X, float Y)> polygon, 
        float unclipRatio)
    {
        if (polygon.Count < 3)
            return polygon;

        var area = CalculatePolygonArea(polygon);
        var perimeter = CalculatePolygonPerimeter(polygon);
        
        if (perimeter <= 0)
            return polygon;

        var distance = area * unclipRatio / perimeter;

        // Clipper2 ClipperOffset works with integer coordinates (Path64)
        // We need to scale our float coordinates to integers, process, then scale back
        const double scale = 1000.0; // Scale factor for precision
        
        // Convert to Path64 (integer coordinates) with scaling
        var path = new Path64(polygon.Select(p => new Point64(
            (long)(p.X * scale),
            (long)(p.Y * scale)
        )));

        // Scale the distance as well
        var scaledDistance = distance * scale;

        // Expand using ClipperOffset
        var co = new ClipperOffset();
        co.AddPath(path, JoinType.Round, EndType.Polygon);
        var solution = new Paths64();
        co.Execute(scaledDistance, solution);

        // Convert back and return first solution
        if (solution.Count > 0 && solution[0].Count >= 3)
        {
            return solution[0].Select(pt => (
                (float)(pt.X / scale),
                (float)(pt.Y / scale)
            )).ToList();
        }

        return polygon;
    }

    /// <summary>
    /// Simplifies a polygon to a quadrilateral (4 corners) by finding the minimum area rotated rectangle
    /// with additional padding to ensure complete text coverage
    /// </summary>
    /// <param name="polygon">Input polygon with any number of points</param>
    /// <param name="paddingFactor">Additional padding factor (default: 1.1 = 10% padding)</param>
    /// <returns>Quadrilateral with exactly 4 corners</returns>
    private static List<(float X, float Y)> SimplifyPolygonToQuadrilateral(
        List<(float X, float Y)> polygon, 
        float topPaddingFactor = 1.0f, float bottomPaddingFactor = 1.0f, float leftPaddingFactor = 1.0f, float rightPaddingFactor = 1.0f)
    {
        if (polygon.Count == 4)
            return polygon;

        if (polygon.Count < 4)
        {
            // Pad with duplicate points if less than 4
            var result = polygon.ToList();
            while (result.Count < 4)
                result.Add(result[result.Count - 1]);
            return result;
        }

        // For polygons with more than 4 points, find the minimum area rotated rectangle
        // Calculate centroid
        var centroidX = polygon.Average(p => p.X);
        var centroidY = polygon.Average(p => p.Y);

        var minArea = float.MaxValue;
        var bestCorners = new List<(float X, float Y)>();

        // Try different rotation angles to find minimum area bounding box
        for (int angleDeg = 0; angleDeg < 180; angleDeg += 5)
        {
            var angleRad = angleDeg * Math.PI / 180.0;
            var cos = Math.Cos(angleRad);
            var sin = Math.Sin(angleRad);

            // Rotate all points around centroid
            var minX = float.MaxValue;
            var maxX = float.MinValue;
            var minY = float.MaxValue;
            var maxY = float.MinValue;

            foreach (var point in polygon)
            {
                var dx = point.X - centroidX;
                var dy = point.Y - centroidY;
                var rotatedX = (float)(dx * cos - dy * sin);
                var rotatedY = (float)(dx * sin + dy * cos);

                minX = Math.Min(minX, rotatedX);
                maxX = Math.Max(maxX, rotatedX);
                minY = Math.Min(minY, rotatedY);
                maxY = Math.Max(maxY, rotatedY);
            }

            var area = (maxX - minX) * (maxY - minY);
            
            if (area < minArea)
            {
                minArea = area;

                // Apply padding factor to expand the box slightly
                var width = maxX - minX;
                var height = maxY - minY;
                var paddingXStart = width * (leftPaddingFactor - 1f) / 2f;
                var paddingXEnd = width * (rightPaddingFactor - 1f) / 2f;
                var paddingYStart = height * (bottomPaddingFactor - 1f) / 2f;
                var paddingYEnd = height * (topPaddingFactor - 1f) / 2f;

                minX -= paddingXStart;
                maxX += paddingXEnd;
                minY -= paddingYStart;
                maxY += paddingYEnd;

                // Rotate corners back to original space
                var corners = new[]
                {
                    (minX, minY), // Top-left
                    (maxX, minY), // Top-right
                    (maxX, maxY), // Bottom-right
                    (minX, maxY)  // Bottom-left
                };

                bestCorners = corners.Select(c =>
                {
                    var x = (float)(c.Item1 * cos + c.Item2 * sin + centroidX);
                    var y = (float)(-c.Item1 * sin + c.Item2 * cos + centroidY);
                    return (x, y);
                }).ToList();
            }
        }

        return bestCorners.Count == 4 ? bestCorners : polygon.Take(4).ToList();
    }

    /// <summary>
    /// Rotates a point around the origin
    /// </summary>
    private static PointF RotatePoint(PointF point, float angleDegrees)
    {
        var angleRad = angleDegrees * Math.PI / 180.0;
        var cos = Math.Cos(angleRad);
        var sin = Math.Sin(angleRad);
        return new PointF(
            (float)(point.X * cos - point.Y * sin),
            (float)(point.X * sin + point.Y * cos)
        );
    }

    /// <summary>
    /// Gets the axis-aligned bounding rectangle of a set of points
    /// </summary>
    private static RectangleF GetAxisAlignedBounds(IEnumerable<PointF> points)
    {
        var pointsList = points.ToList();
        if (pointsList.Count == 0)
            return RectangleF.Empty;

        var minX = pointsList.Min(p => p.X);
        var maxX = pointsList.Max(p => p.X);
        var minY = pointsList.Min(p => p.Y);
        var maxY = pointsList.Max(p => p.Y);
        return new RectangleF(minX, minY, maxX - minX, maxY - minY);
    }

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
    /// <param name="mergeBoxes">Enable box merging for overlapping or nearby boxes (default: false)</param>
    /// <param name="mergeDistanceThreshold">Maximum distance between boxes to merge, as ratio of box height (default: 0.5)</param>
    /// <param name="mergeOverlapThreshold">Minimum IOU for merging overlapping boxes (default: 0.1)</param>
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
        float iouThreshold = 0.3f,
        bool mergeBoxes = false,
        float mergeDistanceThreshold = 0.5f,
        float mergeOverlapThreshold = 0.1f)
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
            bool dimensionsFound = false;
            
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
            
            if (!dimensionsFound)
            {
                var targetAspectRatio = (float)paddedWidth / paddedHeight;
                var bestMatch = (width: mapWidth, height: mapHeight, diff: float.MaxValue);
                
                for (int w = 10; w <= paddedWidth; w++)
                {
                    if (output.Length % w == 0)
                    {
                        int h = output.Length / w;
                        var aspectRatio = (float)w / h;
                        var aspectDiff = Math.Abs(aspectRatio - targetAspectRatio);
                        
                        if (aspectDiff < bestMatch.diff)
                        {
                            bestMatch = (w, h, aspectDiff);
                        }
                    }
                }
                
                if (bestMatch.diff < 0.15f)
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

        // Apply dilation to fill small gaps
        binaryMap = DilateBinaryMap(binaryMap, kernelSize: 2);
        Console.WriteLine($"  Applied dilation with 2x2 kernel");

        // Extract contours
        var contours = ExtractContours(binaryMap);
        Console.WriteLine($"  Found {contours.Count} contours");

        // Calculate proper coordinate scaling
        var mapToResizedScaleX = (float)resizedWidth / mapWidth;
        var mapToResizedScaleY = (float)resizedHeight / mapHeight;
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

            // Find minimum area rotated rectangle
            var minRect = MinimumAreaRectangle.FromContour(contour);
            var rectCorners = minRect.GetCornerPoints();

            Console.WriteLine($"  Contour {boxIndex}: {contour.Count} pts, minAreaRect center=({minRect.Center.X:F1},{minRect.Center.Y:F1}), size={minRect.Size.Width:F1}x{minRect.Size.Height:F1}, angle={minRect.Angle:F1}°");

            // Filter very small detections
            if (minRect.Size.Width < 2 || minRect.Size.Height < 2)
            {
                Console.WriteLine($"    -> SKIPPED (too small)");
                boxIndex++;
                continue;
            }

            // Calculate confidence
            var confidence = CalculateConfidence(output, contour, mapWidth);

            // Scale corners to resized space
            var scaledCorners = rectCorners.Select(p => new PointF(
                p.X * mapToResizedScaleX,
                p.Y * mapToResizedScaleY
            )).ToList();

            // Convert to list of tuples for expansion
            var polygon = scaledCorners.Select(p => (p.X, p.Y)).ToList();

            // Expand using Clipper2
            var expandedPolygon = ExpandPolygonWithClipper(polygon, unclipRatio);

            // Scale to original image space
            var finalPolygon = expandedPolygon.Select(p => (
                p.X * resizedToOriginalScaleX,
                p.Y * resizedToOriginalScaleY
            )).ToList();

            // Clamp to image bounds
            var clampedPolygon = finalPolygon.Select(p => (
                X: Math.Max(0f, Math.Min((float)originalSize.Width, p.Item1)),
                Y: Math.Max(0f, Math.Min((float)originalSize.Height, p.Item2))
            )).ToList();

            // Simplify to exactly 4 corners for BoundingBox compatibility
            var quadrilateral = SimplifyPolygonToQuadrilateral(clampedPolygon, rightPaddingFactor:1.35f);

            // Convert to array format for BoundingBox
            var points = quadrilateral.Select(p => new[] { p.X, p.Y }).ToArray();

            Console.WriteLine($"    -> Expanded polygon: {expandedPolygon.Count} points -> simplified to 4 corners, conf={confidence:F3}");

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
        
        // Apply box merging if enabled
        if (mergeBoxes && boxes.Count > 0)
        {
            var boxesBeforeMerge = boxes.Count;
            boxes = MergeNearbyBoxes(boxes, mergeDistanceThreshold, mergeOverlapThreshold);
            Console.WriteLine($"[POST-PROCESS] Box Merging: {boxesBeforeMerge} -> {boxes.Count} boxes (distance={mergeDistanceThreshold}, overlap={mergeOverlapThreshold})");
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

    /// <summary>
    /// Merges nearby or overlapping bounding boxes
    /// </summary>
    /// <param name="boxes">List of bounding boxes</param>
    /// <param name="distanceThreshold">Maximum distance between boxes to merge (as ratio of box height)</param>
    /// <param name="overlapThreshold">Minimum IOU for merging overlapping boxes</param>
    /// <returns>List of merged bounding boxes</returns>
    private static List<BoundingBox> MergeNearbyBoxes(
        List<BoundingBox> boxes, 
        float distanceThreshold, 
        float overlapThreshold)
    {
        if (boxes.Count <= 1)
            return boxes;

        // Create a union-find structure to track which boxes should be merged
        var parent = Enumerable.Range(0, boxes.Count).ToArray();
        
        int Find(int x)
        {
            if (parent[x] != x)
                parent[x] = Find(parent[x]);
            return parent[x];
        }

        void Union(int x, int y)
        {
            var rootX = Find(x);
            var rootY = Find(y);
            if (rootX != rootY)
                parent[rootX] = rootY;
        }

        // Check all pairs of boxes
        for (int i = 0; i < boxes.Count; i++)
        {
            for (int j = i + 1; j < boxes.Count; j++)
            {
                if (ShouldMergeBoxes(boxes[i], boxes[j], distanceThreshold, overlapThreshold))
                {
                    Union(i, j);
                    Console.WriteLine($"  Merge: Merging box {i} with box {j}");
                }
            }
        }

        // Group boxes by their root parent
        var groups = new Dictionary<int, List<int>>();
        for (int i = 0; i < boxes.Count; i++)
        {
            var root = Find(i);
            if (!groups.ContainsKey(root))
                groups[root] = new List<int>();
            groups[root].Add(i);
        }

        // Merge boxes in each group
        var mergedBoxes = new List<BoundingBox>();
        foreach (var group in groups.Values)
        {
            if (group.Count == 1)
            {
                // Single box, no merging needed
                mergedBoxes.Add(boxes[group[0]]);
            }
            else
            {
                // Merge multiple boxes
                var boxesToMerge = group.Select(idx => boxes[idx]).ToList();
                var merged = MergeBoxGroup(boxesToMerge);
                mergedBoxes.Add(merged);
                Console.WriteLine($"  Merge: Created merged box from {group.Count} boxes");
            }
        }

        return mergedBoxes;
    }

    /// <summary>
    /// Determines if two boxes should be merged based on distance and overlap
    /// </summary>
    private static bool ShouldMergeBoxes(
        BoundingBox box1, 
        BoundingBox box2, 
        float distanceThreshold, 
        float overlapThreshold)
    {
        // Check for overlap
        var iou = CalculateIOU(box1, box2);
        if (iou >= overlapThreshold)
            return true;

        // Check for proximity
        var distance = CalculateBoxDistance(box1, box2);
        var avgHeight = (GetBoxHeight(box1) + GetBoxHeight(box2)) / 2f;
        var normalizedDistance = distance / avgHeight;

        if (normalizedDistance <= distanceThreshold)
        {
            // Additional check: boxes should be roughly aligned horizontally or vertically
            var alignment = GetBoxAlignment(box1, box2);
            return alignment > 0.3f; // 30% overlap in horizontal or vertical direction
        }

        return false;
    }

    /// <summary>
    /// Calculates the minimum distance between two bounding boxes
    /// </summary>
    private static float CalculateBoxDistance(BoundingBox box1, BoundingBox box2)
    {
        var box1MinX = box1.Points.Min(p => p.X);
        var box1MaxX = box1.Points.Max(p => p.X);
        var box1MinY = box1.Points.Min(p => p.Y);
        var box1MaxY = box1.Points.Max(p => p.Y);

        var box2MinX = box2.Points.Min(p => p.X);
        var box2MaxX = box2.Points.Max(p => p.X);
        var box2MinY = box2.Points.Min(p => p.Y);
        var box2MaxY = box2.Points.Max(p => p.Y);

        // Calculate horizontal and vertical gaps
        float horizontalGap;
        if (box1MaxX < box2MinX)
            horizontalGap = box2MinX - box1MaxX;
        else if (box2MaxX < box1MinX)
            horizontalGap = box1MinX - box2MaxX;
        else
            horizontalGap = 0; // Boxes overlap horizontally

        float verticalGap;
        if (box1MaxY < box2MinY)
            verticalGap = box2MinY - box1MaxY;
        else if (box2MaxY < box1MinY)
            verticalGap = box1MinY - box2MaxY;
        else
            verticalGap = 0; // Boxes overlap vertically

        // Euclidean distance of gaps
        return (float)Math.Sqrt(horizontalGap * horizontalGap + verticalGap * verticalGap);
    }

    /// <summary>
    /// Calculates the alignment between two boxes (0-1, where 1 is perfect alignment)
    /// </summary>
    private static float GetBoxAlignment(BoundingBox box1, BoundingBox box2)
    {
        var box1MinX = box1.Points.Min(p => p.X);
        var box1MaxX = box1.Points.Max(p => p.X);
        var box1MinY = box1.Points.Min(p => p.Y);
        var box1MaxY = box1.Points.Max(p => p.Y);

        var box2MinX = box2.Points.Min(p => p.X);
        var box2MaxX = box2.Points.Max(p => p.X);
        var box2MinY = box2.Points.Min(p => p.Y);
        var box2MaxY = box2.Points.Max(p => p.Y);

        // Calculate horizontal overlap
        var horizontalOverlap = Math.Max(0, Math.Min(box1MaxX, box2MaxX) - Math.Max(box1MinX, box2MinX));
        var horizontalUnion = Math.Max(box1MaxX, box2MaxX) - Math.Min(box1MinX, box2MinX);
        var horizontalAlignment = horizontalUnion > 0 ? horizontalOverlap / horizontalUnion : 0;

        // Calculate vertical overlap
        var verticalOverlap = Math.Max(0, Math.Min(box1MaxY, box2MaxY) - Math.Max(box1MinY, box2MinY));
        var verticalUnion = Math.Max(box1MaxY, box2MaxY) - Math.Min(box1MinY, box2MinY);
        var verticalAlignment = verticalUnion > 0 ? verticalOverlap / verticalUnion : 0;

        // Return the best alignment
        return Math.Max(horizontalAlignment, verticalAlignment);
    }

    /// <summary>
    /// Gets the height of a bounding box
    /// </summary>
    private static float GetBoxHeight(BoundingBox box)
    {
        var minY = box.Points.Min(p => p.Y);
        var maxY = box.Points.Max(p => p.Y);
        return maxY - minY;
    }

    /// <summary>
    /// Merges a group of boxes into a single box
    /// </summary>
    private static BoundingBox MergeBoxGroup(List<BoundingBox> boxes)
    {
        // Collect all points from all boxes
        var allPoints = boxes.SelectMany(b => b.Points).ToList();

        // Find the bounding rectangle
        var minX = allPoints.Min(p => p.X);
        var maxX = allPoints.Max(p => p.X);
        var minY = allPoints.Min(p => p.Y);
        var maxY = allPoints.Max(p => p.Y);

        // Use the highest confidence
        var maxConfidence = boxes.Max(b => b.Confidence);

        // Create merged box with 4 corners
        var mergedPoints = new[]
        {
            new[] { minX, minY },
            new[] { maxX, minY },
            new[] { maxX, maxY },
            new[] { minX, maxY }
        };

        return new BoundingBox(mergedPoints, maxConfidence);
    }
}
