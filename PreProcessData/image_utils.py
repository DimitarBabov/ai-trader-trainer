# File: image_utils.py
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def price_to_pixel(value, min_price, max_price, height):
    """Convert a price to a pixel position based on image height and price range."""
    scale = (max_price - min_price) / (height - 1)
    return height - int((value - min_price) / scale) - 1

def create_candlestick_with_regression_image(data, height=224, candlestick_width=3, spacing=0, blur=False, blur_radius = 1, draw_regression_lines=True, color_candles=True):
    """Create a candlestick image with bull and bear candles in different colors, and draw regression lines below."""
    num_candlesticks = len(data)
    min_price = data[['Low']].min().min()
    max_price = data[['High']].max().max()

    # Calculate price change using (close[last candle] - open[first candle]) / open[first candle]
    first_open_price = data['Open'].iloc[0]
    last_close_price = data['Close'].iloc[-1]
    price_change = (last_close_price - first_open_price) / first_open_price

    total_width = (candlestick_width + spacing) * num_candlesticks

    # Render the image in RGB mode with black background
    image = Image.new('RGB', (total_width, height), (0, 0, 0))  # Black background
    draw = ImageDraw.Draw(image)

    # Variable to accumulate the colored pixels (candlestick area)
    sum_colored_pixels = 0

    # Draw the candlesticks
    for i, (index, row) in enumerate(data.iterrows()):
        open_price, high_price, low_price, close_price = row['Open'], row['High'], row['Low'], row['Close']

        # Convert prices to pixel positions for the upper half of the image
        open_pixel = price_to_pixel(open_price, min_price, max_price, height)
        close_pixel = price_to_pixel(close_price, min_price, max_price, height)
        high_pixel = price_to_pixel(high_price, min_price, max_price, height)
        low_pixel = price_to_pixel(low_price, min_price, max_price, height)

        # Determine candle color: green for bull (close > open), red for bear (close < open)
        if color_candles:
            color = (0, 255, 0) if close_price > open_price else (255, 0, 0)
        else:
            color = (255, 255, 255)  # White for all candles if color_candles is False

        # Calculate x position
        x_start = i * (candlestick_width + spacing)

        # Draw wick
        wick_x = x_start + candlestick_width // 2
        draw.line((wick_x, high_pixel, wick_x, low_pixel), fill=color)

        # Draw body
        top_pixel = min(open_pixel, close_pixel)
        bottom_pixel = max(open_pixel, close_pixel)
        draw.rectangle([x_start, top_pixel, x_start + candlestick_width - 1, bottom_pixel], fill=color)

        # Calculate and accumulate the area of the candlestick body
        candlestick_height = abs(open_pixel - close_pixel) + 1
        candlestick_area = candlestick_height * candlestick_width
        sum_colored_pixels += candlestick_area

    # Calculate total pixels area of the image
    total_pixels_area = total_width * height

    # Calculate the ratio of colored pixels to the total area
    colored_pixel_ratio = sum_colored_pixels / total_pixels_area

    # Prepare data for regression calculation by treating each (Open, High, Low, Close) as separate data points
    x_values = []
    y_values = []

    for i in range(num_candlesticks):
        # Four data points per candlestick
        x_values.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
        y_values.extend([data['Open'].iloc[i], data['High'].iloc[i], data['Low'].iloc[i], data['Close'].iloc[i]])

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Calculate regression for all data
    coefficients = np.polyfit(x_values, y_values, 1)
    regression_line = np.poly1d(coefficients)
    slope = coefficients[0]  # Slope of the regression line

    # Calculate the standard deviation around the regression line
    regression_y = regression_line(x_values)
    std_dev = np.std(y_values - regression_y)

    # Calculate the maximum deviation for parallel lines
    max_deviation = max(abs(y_values - regression_y))

    # Scale max deviation to pixels
    max_deviation_scaled = price_to_pixel(max_price - max_deviation, min_price, max_price, height) - price_to_pixel(max_price, min_price, max_price, height)

    # Create upper and lower regression lines
    upper_regression_line = regression_line + max_deviation
    lower_regression_line = regression_line - max_deviation

    # First half regression
    x_values_first = x_values[:len(x_values) // 2]
    y_values_first = y_values[:len(y_values) // 2]
    coefficients_first = np.polyfit(x_values_first, y_values_first, 1)
    regression_line_first = np.poly1d(coefficients_first)
    slope_first = coefficients_first[0]  # Slope of the first regression line

    # Second half regression
    x_values_second = x_values[len(x_values) // 2:]
    y_values_second = y_values[len(y_values) // 2:]
    coefficients_second = np.polyfit(x_values_second, y_values_second, 1)
    regression_line_second = np.poly1d(coefficients_second)
    slope_second = coefficients_second[0]  # Slope of the second regression line

    # Last 4 candles regression
    num_last_candles = 4
    x_values_last = x_values[-num_last_candles * 4:]
    y_values_last = y_values[-num_last_candles * 4:]
    coefficients_last = np.polyfit(x_values_last, y_values_last, 1)
    regression_line_last = np.poly1d(coefficients_last)
    slope_last = coefficients_last[0]  # Slope of the third regression line

    # Normalize y values
    min_y = min(y_values_first.min(), y_values_second.min(), y_values_last.min())
    max_y = max(y_values_first.max(), y_values_second.max(), y_values_last.max())

    def normalize_y(value, min_value, max_value, height):
        """Normalize y values"""
        scale = (max_value - min_value) / (height - 1)
        return height - int((value - min_value) / scale) - 1

    if draw_regression_lines:
        # Draw regression line for all data in the window
        for x in x_values:
            y = regression_line(x)
            y_pixel = normalize_y(y, min_price, max_price, height)
            x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
            if 0 <= y_pixel < height:
                draw.point((x_pixel, y_pixel), fill=(128, 128, 128))  # Draw in gray (128) for visibility
        # Draw upper regression line
        for x in x_values:
            y = upper_regression_line(x)
            y_pixel = price_to_pixel(y, min_price, max_price, height)
            x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
            if 0 <= y_pixel < height:
                draw.point((x_pixel, y_pixel), fill=(200, 200, 200))

        # Draw lower regression line
        for x in x_values:
            y = lower_regression_line(x)
            y_pixel = price_to_pixel(y, min_price, max_price, height)
            x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
            if 0 <= y_pixel < height:
                draw.point((x_pixel, y_pixel), fill=(200, 200, 200))

        # Draw first regression line
        for x in x_values_first:
            y = regression_line_first(x)
            y_pixel = normalize_y(y, min_y, max_y, height)
            x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
            if 0 <= y_pixel < height:
                draw.point((x_pixel, y_pixel), fill=(128, 128, 128))  # Draw in gray (128) for visibility

        # Draw second regression line
        for x in x_values_second:
            y = regression_line_second(x)
            y_pixel = normalize_y(y, min_y, max_y, height)
            x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
            if 0 <= y_pixel < height:
                draw.point((x_pixel, y_pixel), fill=(128, 128, 128))  # Draw in gray (128) for visibility

        # Draw last 4 candles regression line
        for x in x_values_last:
            y = regression_line_last(x)
            y_pixel = normalize_y(y, min_y, max_y, height)
            x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
            if 0 <= y_pixel < height:
                draw.point((x_pixel, y_pixel), fill=(200, 200, 200))  # Draw in light gray (200) for visibility

    if blur:
        # Apply Gaussian blur
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return image, slope_first, slope_second, slope_last, slope, std_dev, colored_pixel_ratio, price_change, max_deviation_scaled
