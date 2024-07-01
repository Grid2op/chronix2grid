import os 

import numpy as np 
import pandas as pd
import seaborn as sns

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def univariate_analysis(df, variable_of_interest,columns_renaming_dict, root_path='.') : 
    """
    This function performs univariate analysis on a given dataframe with respect to a 
    specific variable of interest. It generates two plots for the variable of interest: 
    an empirical cumulative distribution function (ECDF) plot and a histogram.

    Parameters:
    - df (pandas.DataFrame): The input dataframe to analyze.
    - variable_of_interest (str): The name of the variable of interest in the dataframe.
    - columns_renaming_dict (dict): A dictionary that maps column names in the dataframe 
        to their renamed versions. It should be in the format {old_name : new_name}.
    - root_path (str, optional): The path to the directory where the figures will be saved. 
        Default is the current directory.

    Returns:
    - None. Two figures are generated and saved in the Figures directory.

    Example:
        If we want to perform univariate analysis on a dataframe df with variable of interest "duration", 
        and we want to save the figures in a directory named "MyAnalysis" within the current directory, 
        we can call the function like this:

        univariate_analysis(df, "duration", {"duration": "Duration"}, root_path="MyAnalysis")
    """
    renamed_variable = columns_renaming_dict[variable_of_interest]
    figures_folder_path = f'Figures/{root_path}/{renamed_variable}/'
    os.makedirs(figures_folder_path, exist_ok=True)
    
    if(df[variable_of_interest].dtype==np.float16):#float16 not supported in plotly
        df[variable_of_interest]=df[variable_of_interest].astype("float32")

    # ecdf 
    fig = px.ecdf(df, variable_of_interest, labels=columns_renaming_dict, 
                  title=f"{renamed_variable} empirical cumulative distribution function")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_ecdf.png"))
    
    # histogram
    fig = px.histogram(df, variable_of_interest, labels=columns_renaming_dict, 
              title=f"{renamed_variable} histogram")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_histogram.png"))
    
def multivariate_analysis(df, variable_of_interest,columns_renaming_dict, root_path='.') : 
    """
    This function performs multivariate analysis on a given dataframe with respect to a 
    specific variable of interest. It generates two plots for the variable of interest. 
    The first plot is a histogram that shows the distribution of the variable of interest 
    per mix. The second plot is a density plot that shows the relationship between the 
    variable of interest and the duration of the mix, but only if the variable of interest 
    is not "duration".

    Parameters:
     - df (pandas.DataFrame): the dataframe to analyze.
     - variable_of_interest (str): the name of the variable of interest in the dataframe.
     - columns_renaming_dict (dict): a dictionary that maps column names in the dataframe 
        to their renamed versions.
     - root_path (str): the path to the directory where the figures will be saved. 
        Default is the current directory.

    Returns:
        None. Two figures are generated and saved in the Figures directory.
    """
    renamed_variable = columns_renaming_dict[variable_of_interest]
    figures_folder_path = f'Figures/{root_path}/{renamed_variable}/'
    os.makedirs(figures_folder_path, exist_ok=True)
    
    # histogram per mix
    fig = px.histogram(df, variable_of_interest, labels=columns_renaming_dict,color="mix",
                       title=f"{renamed_variable} histogram per mix")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_histogram_per_mix.png"))
    
    # density plot 
    if variable_of_interest != "duration":
        fig = px.density_contour(df, x=variable_of_interest, y="duration",
                                 labels=columns_renaming_dict, 
                                 color="mix",
                           title=f"{renamed_variable} density plot per mix")
        fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_density_plot_per_mix.png"))
        

def decomposition_analysis(df, variable_of_interest,columns_renaming_dict, root_path='.') : 
    """
    This function performs decomposition analysis on a given dataframe with respect to a 
    specific variable of interest. It generates five boxplots for the variable of interest, 
    each showing how the variable varies with respect to a different categorical variable in 
    the dataset: mix, overload line name, month, hour of day, and day of the week.

    Parameters:
    - df (pandas.DataFrame): The input dataframe to analyze.
    - variable_of_interest (str): The name of the variable of interest in the dataframe.
    - columns_renaming_dict (dict): A dictionary that maps column names in the dataframe to 
        their renamed versions. It should be in the format {old_name : new_name}.
    - root_path (str, optional): The path to the directory where the figures will be saved. 
        Default is the current directory.

    Returns:
    - None. Five figures are generated and saved in the Figures directory.

    Example:
        If we want to perform decomposition analysis on a dataframe df with variable of interest "maxDepths", 
        and we want to save the figures in a directory named "MyAnalysis" within the current directory, 
        we can call the function like this:

        decomposition_analysis(df, "maxDepths", {"maxDepths": "Rho max"}, root_path="MyAnalysis")
    
    """
    renamed_variable = columns_renaming_dict[variable_of_interest]
    figures_folder_path = f'Figures/{root_path}/{renamed_variable}/'
    os.makedirs(figures_folder_path, exist_ok=True)
    
    # boxplot per mix 
    fig = px.box(df, x="mix", y=variable_of_interest, 
           labels=columns_renaming_dict,
           title=f"{renamed_variable} per mix",
           color="mix")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_boxplot_per_mix.png"))
    
    
    # boxplot per line and mix
    fig = px.box(df, x="overload_line_name", y=variable_of_interest, 
           labels=columns_renaming_dict,
           title=f"{renamed_variable} separated by line and by mix",
           color="mix")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_boxplot_per_line_and_per_mix.png"))
    
    
    # boxplot per month and mix
    fig = px.box(df, x="month", y=variable_of_interest, 
           labels=columns_renaming_dict,
           title=f"{renamed_variable} separated by month and by mix",
           color="mix")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_boxplot_per_month_and_per_mix.png"))
    
    
    # boxplot per hour_of_day and mix
    fig = px.box(df, x="hour_of_day", y=variable_of_interest, 
           labels=columns_renaming_dict,
           title=f"{renamed_variable} separated by hour and by mix",
           color="mix")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_boxplot_per_hour_and_per_mix.png"))
    
    # boxplot per day_of_week and mix
    fig = px.box(df, x="day_of_week", y=variable_of_interest, 
           labels=columns_renaming_dict,
           title=f"{renamed_variable} separated by day of the week and by mix",
           color="mix")
    fig.write_image(os.path.join(figures_folder_path, f"{renamed_variable}_boxplot_per_dow_and_per_mix.png"))
    
    
def aggregated_operation_analysis(df, variable_of_interest, columns_renaming_dict, operation='count', 
                                  operation_full_name="Number of overflows", root_path="DoNothing"):
    """
    The aggregated_operation_analysis function takes in six parameters:
     - df: a pandas DataFrame object, representing the dataset to be analyzed.
     - variable_of_interest: a string, representing the column of interest in the dataset.
     - columns_renaming_dict: a dictionary, representing the renaming of columns in the dataset.
     - operation: a string, representing the type of operation to be performed on the dataset 
         (default value: 'count').
     - operation_full_name: a string, representing the full name of the operation performed 
         (default value: "Number of overflows").
     - root_path: a string, representing the root path where the resulting figures will be saved 
         (default value: "DoNothing").

    The function returns nothing, but generates and saves a series of figures based on the input dataset and parameters.

    Note: This function assumes that the plotly and os modules have been imported beforehand.
    """
    renamed_variable = columns_renaming_dict[variable_of_interest]
    figures_folder_path = f'Figures/{root_path}/{renamed_variable}_{operation}'
    os.makedirs(figures_folder_path, exist_ok=True)
    
    fig = px.bar(getattr(df.groupby(['month', 'mix']), operation)().reset_index(), 
                 x="month", 
                 y=variable_of_interest, 
                 color="mix", 
                 labels={"mix": "Mix", variable_of_interest: operation_full_name, 'month': "Mois"},
                 title=f"{operation_full_name} per month per mix")
    
    fig.write_image(os.path.join(figures_folder_path, 
                                 f"{operation_full_name.replace(' ', '_')}_barplot_for_each_month_and_mix.png"))


    fig = px.bar(getattr(df.groupby(['month', 'mix']), operation)().reset_index(), 
           x="mix", 
           y=variable_of_interest, 
           color="month", 
           labels={"mix": "Mix", variable_of_interest: operation_full_name, 'month': "Mois"},
           title=f"{operation_full_name}  per mix per month")
    fig.write_image(os.path.join(figures_folder_path, 
                                 f"{operation_full_name.replace(' ', '_')}_barplot_for_each_mix_and_month.png"))

    fig = px.bar(getattr(df.groupby(['overload_line_name', 'mix']), operation)().reset_index().sort_values("mix"), 
           x="mix", 
           y=variable_of_interest, 
           color="overload_line_name", 
           title=f"{operation_full_name} per mix and lignes", 
           labels={"mix": "Mix", variable_of_interest: operation_full_name, 'overload_line_name': "Ligne"})
    fig.write_image(os.path.join(figures_folder_path, 
                                 f"{operation_full_name.replace(' ', '_')}_barplot_for_each_line_mix.png"))

    fig = px.sunburst(getattr(df.groupby(['overload_line_name', 'mix']), operation)().reset_index(), 
                path=["mix", 'overload_line_name'], 
                values=variable_of_interest,
                labels={"mix": "Mix", variable_of_interest: operation_full_name, 'overload_line_name': "Ligne"},
                title=f"{operation_full_name} sunburst per mix and line",
                color=variable_of_interest, hover_data=[variable_of_interest])
    fig.write_image(os.path.join(figures_folder_path, 
                                 f"{operation_full_name.replace(' ', '_')}_sunburst_for_each_mix_and_line.png"))

    fig = px.treemap(getattr(df.groupby(['overload_line_name', 'mix', 'month']), operation)().reset_index(), 
               path=["mix", "month", 'overload_line_name'], values=variable_of_interest,
               labels={"mix": "Mix", variable_of_interest: operation_full_name, 'overload_line_name': "Ligne"},
                title=f"{operation_full_name} treemap per mix, month and line",
               color=variable_of_interest, hover_data=[variable_of_interest])
    fig.write_image(os.path.join(figures_folder_path, 
                                 f"{operation_full_name.replace(' ', '_')}_treemap_for_each_mix_month_line.png"))

    fig = px.treemap(getattr(df.groupby(['overload_line_name', 'mix', 'month']), operation)().reset_index(), 
               path=["mix", 'overload_line_name', "month"], values=variable_of_interest,
               labels={"mix": "Mix", variable_of_interest: operation_full_name, 'overload_line_name': "Ligne"},
                title=f"{operation_full_name} treemap per mix, line and month",
               color=variable_of_interest, hover_data=[variable_of_interest])
    fig.write_image(os.path.join(figures_folder_path, 
                                 f"{operation_full_name.replace(' ', '_')}_treemap_for_each_mix_line_month.png"))
    
def create_heatmaps_nb_overloads(df,columns_renaming_dict, root_path='.') : 
    """
    Creates density heatmaps and pivot tables to visualize the number of overloads for different categories and mixes.

    Parameters:

     - df (pandas.DataFrame): input DataFrame containing the data for creating heatmaps and pivot tables.
     - columns_renaming_dict (dict): dictionary containing the renaming of the columns in the DataFrame.
     - root_path (str, optional): the root path for saving the figures. Default is '.'.

    Returns:
    None
    """
    figures_folder_path = f'Figures/{root_path}/other/'
    os.makedirs(figures_folder_path, exist_ok=True)
    
    fig = px.density_heatmap(df,
                       y= "mix", 
                       x="month",
                       title = f'Number of overload for each month and each mix')
    fig.write_image(os.path.join(figures_folder_path, f"nb_overloads_heatmap_for_each_month_and_mix.png"))

    
    # heatmap 
    fig = px.density_heatmap(df,
                       y= "mix", 
                       x="overload_line_name",
                       title = f'Number of overload for each line and each mix')
    fig.write_image(os.path.join(figures_folder_path, f"nb_overloads_heatmap_for_each_line_and_mix.png"))
    

    fig = px.density_heatmap(df,
                       y= "mix", 
                       x="scenarios",
                       title = f'Number of overload for each line and each mix')
    fig.write_image(os.path.join(figures_folder_path, f"nb_overloads_heatmap_for_each_scenario_and_mix.png"))

    
    
    
    fig = px.density_heatmap(df,
                       y= "overload_line_name", 
                       x="month",
                       title = f'Number of overload for each month and each mix')
    fig.write_image(os.path.join(figures_folder_path, f"nb_overloads_heatmap_for_each_line_per_month.png"))
    
    
    for agg_name, agg_func in {'min':np.min, 'max': np.max, 'mean': np.mean}.items():
        for var in ['duration', 'maxDepths']:
            table_agent_overload_name = pd.pivot_table(df, values=var, index=['overload_line_name'],
                            columns=['mix'], aggfunc=agg_func,fill_value=0)
            fig=px.imshow(table_agent_overload_name, 
                          title=f'{var}_{agg_name}_heatmap_for_each_line_and_mix')
            fig.write_image(os.path.join(figures_folder_path, f"{var}_{agg_name}_heatmap_for_each_line_and_mix.png"))


            table_agent_scenario = pd.pivot_table(df, values=var, index=['scenarios'],
                            columns=['mix'], aggfunc=agg_func,fill_value=0)
            fig=px.imshow(table_agent_scenario, 
                          title=f'{var}_{agg_name}_heatmap_for_each_scenario_and_mix')
            fig.write_image(os.path.join(figures_folder_path, f"{var}_{agg_name}_heatmap_for_each_scenario_and_mix.png"))


            table_month_line_overload = pd.pivot_table(df, values=var, index=['month'],
                            columns=['mix'], aggfunc=agg_func,fill_value=0)
            fig=px.imshow(table_month_line_overload,
                          title=f'{var}_{agg_name}_heatmap_for_each_month_and_mix')
            fig.write_image(os.path.join(figures_folder_path, f"{var}_{agg_name}_heatmap_for_each_month_and_mix.png"))

            table_dow_line_overload = pd.pivot_table(df, values=var, index=['day_of_week'],
                            columns=['mix'], aggfunc=agg_func,fill_value=0)
            fig=px.imshow(table_dow_line_overload,
                          title=f'{var}_{agg_name}_heatmap_for_each_day_of_week_and_mix')
            fig.write_image(os.path.join(figures_folder_path, f"{var}_{agg_name}_heatmap_for_each_day_of_week_and_mix.png"))


            table_hod_line_overload = pd.pivot_table(df, values=var, index=['hour_of_day'],
                            columns=['mix'], aggfunc=agg_func,fill_value=0)
            fig=px.imshow(table_hod_line_overload,
                          title=f'{var}_{agg_name}_heatmap_for_each_hour_and_mix')
            fig.write_image(os.path.join(figures_folder_path, f"{var}_{agg_name}_heatmap_for_each_hour_of_day_and_mix.png"))
    
    
def custom_plots(df,columns_renaming_dict, root_path='.'): 
    """
    A function that generates custom plots for a given pandas DataFrame.

    Args:
    - df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    - columns_renaming_dict (dict): A dictionary containing the column renaming information. The keys are the current
    - column names in the DataFrame, and the values are the new column names.
    - root_path (str): The path to the directory where the output figures will be saved. Default is the current directory.

    Returns:
    None.

    The function generates joint plots of 'duration' and 'maxDepths' for each unique value in the 'mix' column of the DataFrame.
    Each plot is saved in the 'Figures/root_path/other/' directory with a file name of the format
    'jointplot_rho_max_vs_duration_mix_X.png',
    where X is the unique value of 'mix' for that plot. Finally, the function creates a 3x2 grid of images, each image
    corresponding to a 'mix' value. The grid is saved in the 'Figures/root_path/other/' directory with the file name
    'jointplot_rho_max_vs_duration_all_mixes.png'.
    """
    figures_folder_path = f'Figures/{root_path}/other/'
    os.makedirs(figures_folder_path, exist_ok=True)

    mixes=df.mix.unique()
    for mix in mixes:
        sns.jointplot(x=df[df.mix==mix].duration, xlim=(df.duration.min(), df.duration.max()),
                      y=df[df.mix==mix].maxDepths,  ylim=(df.maxDepths.min(), df.maxDepths.max()),
                      kind="hex", bins="log", hue_norm="log")
        plt.suptitle(f"Joint plot of rho max vs duration for mix {mix}")
        plt.xlabel('Durée')
        plt.ylabel('Rho max')
        plt.savefig(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_{mix}.png")
        plt.show()
        
        
    
    f, axarr = plt.subplots(int(np.floor(len(mixes)/2)), 2, figsize=(25, 30))
    is_arr_one_dimensional=(len(axarr.shape)==1)
    for i,mix in enumerate(mixes):
        if(is_arr_one_dimensional):
            axarr[i].imshow(mpimg.imread(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_{mixes[i]}.png"))
        else:    
            axarr[int(np.floor(i/2)),i%2].imshow(mpimg.imread(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_{mixes[i]}.png"))
        #axarr[0,1].imshow(mpimg.imread(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_{mixes[1]}.png"))
        #axarr[1,0].imshow(mpimg.imread(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_{mixes[2]}.png"))
        #axarr[1,1].imshow(mpimg.imread(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_{mixes[3]}.png"))
        #axarr[2,0].imshow(mpimg.imread(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_{mixes[4]}.png"))

    # turn off x and y axis
    [ax.set_axis_off() for ax in axarr.ravel()]

    plt.tight_layout()
    plt.savefig(f"Figures/{root_path}/other/jointplot_rho_max_vs_duration_all_mixes.png")
    plt.show()