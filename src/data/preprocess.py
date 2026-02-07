from src.utils.logger import get_logger


def preprocess_data(df, output_path):
    """
    Preprocess the input DataFrame by handling missing values and duplicates.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    logger = get_logger(__name__)
    logger.info("Starting data preprocessing...")
    df = df.drop_duplicates().reset_index(drop=True)

    df.to_csv(output_path, index=False)
    logger.info("Preprocessed data saved to %s", output_path)
    return df