import numpy as np
import pandas as pd


def generate_dataset(n):
    np.random.seed(42)  # Setting seed for reproducibility

    # Generating X values with limited range and decimal points
    X1 = np.round(np.random.uniform(-21, 50, n), 4)
    X2 = np.round(np.random.uniform(1, 13, n), 4)
    X3 = np.round(np.random.uniform(1, 17, n), 4)
    X4 = np.round(np.random.uniform(-12, 25, n), 4)
    X5 = np.round(np.random.uniform(-51, 11, n), 4)
    X6 = np.round(np.random.uniform(0, 25, n), 4)
    X7 = np.round(np.random.uniform(-11, 9, n), 4)
    X8 = np.round(np.random.uniform(-12, 27, n), 4)

    """
    Y = 2 * X1 + 3 * X2 ** 1.5 - 1.5 * X3 ** 0.5 + 4 * X4 \
        + 2.56 * X2 / X5 - 5.4 * X1 ** 3 * X6 - X1 * X7 ** 2 - X6 * X9 * X10 ** 0.5 \
        - 1.42 * X11 / X8 ** 2 + X3 ** 2 / X5 + X2 ** 2 * X8 - X4 * X10 - X7 ** 2 / X9 \
        + np.round(np.random.randn(n), 4)   
    """
    # Generating Y values based on the specified formula with added noise
    Y = (2 * X1) - (3 * X2) - (1.5 * (X3 ** 2)) + (4 * X4) \
        + (2.5 * X2 * X5) - (5 * X8 * X6) - (X1 * X7) - (X4 * (X1 ** 2)) \
        - (1.5 * X7) + ((X3 ** 2) * X5) + (X2 ** 2) - (X4 * X8) - (X7 * X3) \
        + np.round(np.random.randn(n), 4)

    # Creating a DataFrame
    data = {
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X4,
        'X6': X4,
        'X7': X4,
        'X8': X4,
        'Y': Y
    }

    df = pd.DataFrame(data)

    # Save DataFrame to a CSV file
    df.to_csv('randGenBigDS1.csv', index=False, float_format='%.4f')  # Limit decimal points to 4
    print(f"Dataset saved as 'randGenBigDS1.csv' in the 'Files' folder.")
    return df


# Generating a DataFrame with 10000 rows and saving it to a CSV file
n = 10000
generated_df = generate_dataset(n)
print(generated_df)
