import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math 

def get_random_patient_sample(df, patient_col, num_samples):
    """
    Get a random sample of X unique patients from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    patient_col (str): The name of the column with patient IDs.
    num_samples (int): The number of unique patients to sample.
    
    Returns:
    pd.DataFrame: A DataFrame containing the sampled patients.
    """
    # Ensure the patient column is unique
    unique_patients = df[patient_col].unique()
    
    # Sample X unique patients
    sampled_patients = pd.Series(unique_patients).sample(n=num_samples, replace=False).tolist()
    
    # Filter the DataFrame to include only the sampled patients
    sampled_df = df[df[patient_col].isin(sampled_patients)]
    
    return sampled_df


def ensure_combinations_in_sample(df, patient_col='RID', dx_col='DX', num_samples=10):
    """
    Get a random sample of patients ensuring at least one patient from each unique combination of diagnostic categories,
    and return the full dataset for these patients.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    patient_col (str): The name of the column with patient IDs.
    dx_col (str): The name of the column with diagnostic categories.
    num_samples (int): The total number of unique patients to sample.
    
    Returns:
    pd.DataFrame: A DataFrame containing the full dataset for the sampled patients.
    """
    # Step 1: Group by patient ID and aggregate unique diagnostic categories
    diagnosis_combinations = df.groupby(patient_col)[dx_col].apply(lambda x: ', '.join(sorted(x.unique()))).reset_index()
    diagnosis_combinations.columns = [patient_col, 'Combination']
    
    # Step 2: Ensure each combination is represented
    unique_combinations = diagnosis_combinations['Combination'].unique()
    samples = []
    
    # Sample one patient for each unique combination
    for combination in unique_combinations:
        patient = diagnosis_combinations[diagnosis_combinations['Combination'] == combination].sample(n=1)
        samples.append(patient)
    
    # Combine samples into one DataFrame
    samples_df = pd.concat(samples).reset_index(drop=True)
    
    # Step 3: Optionally, add more patients to the sample (excluding already sampled)
    remaining_df = diagnosis_combinations[~diagnosis_combinations[patient_col].isin(samples_df[patient_col])]
    
    # Calculate number of additional samples needed
    additional_samples_needed = max(0, num_samples - len(samples_df))
    
    if additional_samples_needed > 0 and not remaining_df.empty:
        # Randomly sample additional patients
        additional_samples = remaining_df.sample(n=min(additional_samples_needed, len(remaining_df)), random_state=1)
        
        # Combine additional samples
        final_sample_df = pd.concat([samples_df, additional_samples]).drop_duplicates().reset_index(drop=True)
    else:
        final_sample_df = samples_df
    
    # Get the full dataset for the sampled patient IDs
    sampled_patient_ids = final_sample_df[patient_col].unique()
    full_sampled_df = df[df[patient_col].isin(sampled_patient_ids)]
    
    return full_sampled_df


def viscode_to_month(viscode):
    """
    Converts VISCODE labels to numerical month values.
    
    Parameters:
    viscode (str): The VISCODE label.
    
    Returns:
    int: The corresponding month value.
    """
    if viscode == 'bl':
        return 0
    elif viscode.startswith('m'):
        return int(viscode[1:])
    else:
        raise ValueError(f"Unknown VISCODE label: {viscode}")


def create_next_viscode(df):
    """
    Creates the NextVISCODE column and fills missing values.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the VISCODE column.
    
    Returns:
    pd.DataFrame: DataFrame with the NextVISCODE column added.
    """
    df = df.copy()  # Ensure we're working with a copy to avoid SettingWithCopyWarning
    df.loc[:, 'NextVISCODE'] = df.groupby('RID')['VISCODE'].shift(-1)
    df.loc[:, 'NextVISCODE'] = df.groupby('RID')['NextVISCODE'].ffill()
    return df


def setup_plot():
    """
    Sets up the plot with predefined handles for the legend.
    
    Returns:
    fig, ax (tuple): Matplotlib figure and axis objects.
    handles (dict): Dictionary of legend handles.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    stage_colors = {'CN': 'skyblue', 'MCI': 'orange', 'Dementia': 'green'}
    handles = {stage: plt.Line2D([0], [0], color=color, lw=4) for stage, color in stage_colors.items()}
    
    return fig, ax, handles


def plot_patient_stages(df):
    """
    Plots a horizontal bar chart for multiple patients' stages based on VISCODE.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing patient stages with columns 'RID', 'assumedDX', 'VISCODE', and 'NextVISCODE'.
    
    Returns:
    None
    """
    # Create NextVISCODE column
    df = create_next_viscode(df)
    
    # Convert VISCODE and NextVISCODE to numerical month values
    df.loc[:, 'StartMonth'] = df['VISCODE'].apply(viscode_to_month)
    df.loc[:, 'EndMonth'] = df['NextVISCODE'].apply(viscode_to_month)
    
    # Define colors for different stages for better visualization
    stage_colors = {'CN': 'skyblue', 'MCI': 'orange', 'Dementia': 'green'}
    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize if needed
    
    # Generate a unique index for each patient for plotting
    df.loc[:, 'PatientIndex'] = pd.Categorical(df['RID']).codes
    
    # Plot each patient's stages as horizontal bars
    for _, row in df.iterrows():
        ax.barh(y=row['PatientIndex'], width=row['EndMonth'] - row['StartMonth'], 
                left=row['StartMonth'], height=0.3, align='center',
                color=stage_colors.get(row['assumedDX'], 'gray'))
    
    # Remove padding around the plot
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    
    # Set y-axis labels to patient IDs
    ax.set_yticks(df['PatientIndex'].unique())
    ax.set_yticklabels(df['RID'].unique())
    
    # Set x-axis labels for months
    ax.set_xticks(sorted(df['StartMonth'].unique()))
    ax.set_xticklabels([f'{x} months' for x in sorted(df['StartMonth'].unique())])
    ax.set_xlabel('Months')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Set axis labels and title
    ax.set_title('Patient History')
    
    # Define legend handles manually
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in stage_colors.values()]
    labels = stage_colors.keys()
    
    # Add a legend
    ax.legend(handles, labels, loc='upper right')
    
    # Display the plot
    plt.show()


def plot_numeric_columns(df):
    """
    Create bar charts for all numeric columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with numeric columns.

    Returns:
    None
    """
    # Get all numeric columns in the DataFrame
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Create a figure with subplots
    num_plots = len(numeric_columns)
    num_cols = 3  # Number of columns in subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over
    
    # Plot each numeric column
    for i, col in enumerate(numeric_columns):
        df[col].value_counts().plot(kind='bar', ax=axes[i], color='skyblue')
        axes[i].set_title(col)
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_categorical_columns(df):
    """
    Create bar charts for all categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with categorical columns.

    Returns:
    None
    """
    # Get all categorical columns in the DataFrame
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Create a figure with subplots
    num_plots = len(categorical_columns)
    num_cols = 3  # Number of columns in subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over
    
    # Plot each categorical column
    for i, col in enumerate(categorical_columns):
        df[col].value_counts().plot(kind='bar', ax=axes[i], color='skyblue')
        axes[i].set_title(col)
        axes[i].set_xlabel('Categories')
        axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_diagnosis_groups(df):
    """
    Separates patients into different diagnosis progression groups.

    Parameters:
    month_six_df (pd.DataFrame): DataFrame filtered for month six records.

    Returns:
    tuple: A tuple containing DataFrames for each diagnosis progression group.
    """
    no_change = df[df['assumedDX'] == df['DX_bl_mapping']]
    cn_mci = df[
        (df['assumedDX'] == 'MCI') & (df['DX_bl_mapping'] == 'CN')
    ]
    mci_ad = df[
        (df['assumedDX'] == 'Dementia') & (df['DX_bl_mapping'] == 'MCI')
    ]
    cn_ad = df[
        (df['assumedDX'] == 'Dementia') & (df['DX_bl_mapping'] == 'CN')
    ]
    return no_change, cn_mci, mci_ad, cn_ad


def calculate_bins(*groups):
    """
    Calculate the number of bins for each group's histogram.

    Parameters:
    *groups (pd.DataFrame): DataFrames for different diagnosis progression groups.

    Returns:
    list: A list of integers representing the number of bins for each group.
    """
    return [int(np.sqrt(group.shape[0])) for group in groups]


def plot_kde_only(column, dist_groups, labels):
    """
    Creates KDE curve plots for the specified column across different diagnosis progression groups.

    Parameters:
    column (str): The column name to plot.
    dist_groups (tuple): Tuple of DataFrames, each representing a diagnosis progression group.
    labels (tuple): Tuple of labels corresponding to each group.
    """
    colors = ['blue', 'yellow', 'green', 'red', 'orange', 'purple']
    alphas = [0.6, 0.5, 0.4, 0.2, 0.2, 0.2]

    plt.rcParams["figure.figsize"] = (14, 6)

    # Plot KDE only
    plt.figure()
    for i in range(len(dist_groups)):
        label = labels[i]
        sns.kdeplot(
            dist_groups[i][column].dropna().values,  # Drop NaNs to avoid issues
            label=label,
            color=colors[i],
            alpha=alphas[i]
        )

    xlabel = (
        column[:-6] + ' Change' if '_delta' in column
        else 'Baseline ' + column[:-3] if '_bl' in column
        else column
    )
    title = (
        column[:-6] + ' Change by Change in Diagnosis'
        if '_delta' in column
        else 'Baseline ' + column[:-3] + '\nby Change in Diagnosis'
        if '_bl' in column
        else column + ' by Change in Diagnosis'
    )
    
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Kernel Density Estimate')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def filter_data_by_time_points(df, time_points):
    """
    Filters the dataset for each specified time point.

    Parameters:
    df (DataFrame): The dataset to filter.
    time_points (list): The list of time points to filter by.

    Returns:
    dict: A dictionary where keys are time points and values are filtered DataFrames.
    """
    return {tp: df[df['VISCODE'] == tp] for tp in time_points}


def count_diagnoses_by_time_point(filtered_data, dx_order):
    """
    Counts the diagnoses for each time point.

    Parameters:
    filtered_data (dict): A dictionary of filtered DataFrames by time point.
    dx_order (list): The list of diagnoses to count in order.

    Returns:
    dict: A dictionary where keys are time points and values are diagnosis counts.
    """
    return {tp: data['DX'].value_counts().reindex(dx_order, fill_value=0) for tp, data in filtered_data.items()}


def determine_grid_size(n):
    """
    Determines the grid size for subplots.

    Parameters:
    n (int): The number of subplots needed.

    Returns:
    tuple: A tuple representing the grid size (rows, cols).
    """
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    return rows, cols


def plot_diagnosis_counts(diagnosis_counts, grid_size, dx_order):
    """
    Plots bar charts of diagnosis counts for each time point.

    Parameters:
    diagnosis_counts (dict): A dictionary of diagnosis counts by time point.
    grid_size (tuple): A tuple representing the grid size (rows, cols).
    dx_order (list): The list of diagnoses to plot in order.
    """
    fig, axes = plt.subplots(*grid_size, figsize=(20, 15), sharey=True)
    axes = axes.flatten()

    for ax, (tp, counts) in zip(axes, diagnosis_counts.items()):
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(f'{tp}')
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for ax in axes[len(diagnosis_counts):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    

def map_diagnoses(df, dx_mapping):
    """
    Maps the diagnoses in the DataFrame based on a provided mapping.

    Parameters:
    df (DataFrame): The dataset containing diagnoses to map.
    dx_mapping (dict): A dictionary mapping old diagnoses to new ones.

    Returns:
    DataFrame: The DataFrame with an additional column for mapped diagnoses.
    """
    df['DX_bl_mapping'] = df['DX_bl'].replace(dx_mapping)
    return df



import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_kde_only(columns, dist_groups, labels):
    """
    Creates KDE curve plots for the specified columns across different diagnosis progression groups.

    Parameters:
    columns (list): List of column names to plot.
    dist_groups (tuple): Tuple of DataFrames, each representing a diagnosis progression group.
    labels (tuple): Tuple of labels corresponding to each group.
    """
    colors = ['blue', 'yellow', 'green', 'red', 'orange', 'purple']
    alphas = [0.6, 0.5, 0.4, 0.2, 0.2, 0.2]

    num_columns = len(columns)
    num_rows = math.ceil(num_columns / 2)  # Number of rows needed, with 2 columns per row

    plt.rcParams["figure.figsize"] = (16, 6 * num_rows)

    fig, axes = plt.subplots(num_rows, 2, figsize=(16, 6 * num_rows))
    axes = axes.flatten()  # Flatten the grid to easily iterate over axes

    for idx, column in enumerate(columns):
        ax = axes[idx]
        for i in range(len(dist_groups)):
            label = labels[i]
            sns.kdeplot(
                dist_groups[i][column].dropna().values,  # Drop NaNs to avoid issues
                label=label,
                color=colors[i],
                alpha=alphas[i],
                ax=ax
            )

        xlabel = (
            column[:-6] + ' Change' if '_delta' in column
            else 'Baseline ' + column[:-3] if '_bl' in column
            else column
        )
        title = (
            column[:-6] + ' Change by Change in Diagnosis'
            if '_delta' in column
            else 'Baseline ' + column[:-3] + '\nby Change in Diagnosis'
            if '_bl' in column
            else column + ' by Change in Diagnosis'
        )

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylabel('Kernel Density Estimate')
        ax.legend(loc='best')

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
