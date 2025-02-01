using System.Text.RegularExpressions;

namespace faceoff.Extensions
{
    public static class RegexExtensions
    {
        private const string topPattern = @"top:\S+;";
        private const string leftPattern = @"left:\S+;";
        private const string widthPattern = @"width:\S+;";
        private const string heightPattern = @"height:\S+;";
        private const string colorPattern = @"outline-color:\S+;";
        public static string ToTrackingStyle(this string trackingStyle, int x, int y,  int width, int height)
        {
            // No face detected. Hide the frame!
            if (x == 0 && y == 0)
            {
                trackingStyle = "display: none;";
                return trackingStyle;
            }
            
            if (trackingStyle == "display: none;")
            {
                trackingStyle = "";
            }
            var top = y - (height / 2);
            var left = x - (width / 2);

            if (!string.IsNullOrEmpty(trackingStyle))
            {
                var topRegex = new Regex(topPattern, RegexOptions.IgnoreCase);
                var leftRegex = new Regex(leftPattern, RegexOptions.IgnoreCase);
                var widthRegex = new Regex(widthPattern, RegexOptions.IgnoreCase);
                var heightRegex = new Regex(heightPattern, RegexOptions.IgnoreCase);

                if (topRegex.IsMatch(trackingStyle))
                {
                    trackingStyle = topRegex.Replace(trackingStyle, $"top:{top}px;");
                }
                else
                {
                    trackingStyle += $"top:{top}px;";
                }

                if (leftRegex.IsMatch(trackingStyle))
                {
                    trackingStyle = leftRegex.Replace(trackingStyle, $"left:{left}px;");
                }
                else
                {
                    trackingStyle += $"left:{left}px;";
                }

                if (widthRegex.IsMatch(trackingStyle))
                {
                    trackingStyle = widthRegex.Replace(trackingStyle, $"width:{width}px;");
                }
                else
                {
                    trackingStyle += $"width:{width}px;";
                }

                if (heightRegex.IsMatch(trackingStyle))
                {
                    trackingStyle = heightRegex.Replace(trackingStyle, $"height:{height}px;");
                }
                else
                {
                    trackingStyle += $"height:{height}px;";
                }
            }
            else
            {
                trackingStyle = $"top:{top}px;left:{left}px;width:{width}px;height:{height}px;";
            }
            return trackingStyle;
        }

        public static string ToColorStyle(this string colorStyle, string color)
        {
            var style = $"outline-color:{color};";

            if (!string.IsNullOrEmpty(colorStyle))
            {
                var regex = new Regex(colorPattern, RegexOptions.IgnoreCase);
                if (regex.IsMatch(colorStyle))
                {
                    colorStyle = regex.Replace(colorStyle, style);
                }
                else
                {
                    colorStyle += style;
                }
            }
            else
            {
                colorStyle = style;
            }


            return colorStyle;
        }
    }
}
