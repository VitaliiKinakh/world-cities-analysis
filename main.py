import geopandas as gpd
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def basic_statistic_values(df):
    min = df.min()
    max = df.max()
    mean = df.mean()
    median = df.median()
    mode = df.mode()
    std = df.std()
    return {"min" : min, "max" : max, "mean" : mean, "median" : median,  "std" : std}


def create_plot(df, col, name, n_bins, y_tic, x_tic):
    first_df = df[df[col] != 0][col]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # Get some statistics
    min_val = np.min(first_df)
    max_val = np.max(first_df)
    mean_val = np.mean(first_df)

    # Create first hist
    n1, bins1, pathes = ax1.hist(first_df, n_bins, histtype="bar", rwidth=0.8, range=[min_val, mean_val])

    # Visualisation
    ax1.set_ylabel(y_tic)
    ax1.set_xlabel(x_tic)
    ax1.set_xticks(bins1)
    ax1.set_xticklabels(bins1, rotation=85)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.set_title(name)
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{}'.format(int(x * 4)) for x in vals])

    # Prepare data for second hist
    second_df = df[df[col] >= mean_val][col]
    # Prepare second bins
    diff = (max_val - mean_val) / n_bins
    m = mean_val
    b = []
    while m <= max_val:
        b.append(m)
        m += diff

    # Create second plot
    n2, bins2, pathes = ax2.hist(second_df, bins=b, histtype="bar", rwidth=0.8)

    # Visualisation
    ax2.set_ylabel(y_tic)
    ax2.set_xlabel(x_tic)
    ax2.set_xticklabels(bins2, rotation=85)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{}'.format(int(x * 0.4)) for x in vals])

    fig.tight_layout()
    plt.savefig(str(name) + ".png", dpi=300)

if __name__ == "__main__":
    cities_db = gpd.read_file("cities.shp")
    cities_db_with_population = pd.read_csv("worldcitiespop.txt", encoding = "ISO-8859-1")

    cities_db = cities_db.to_crs({'init': 'epsg:3857'})

    # Get cities area
    cities_db["AREA"] = cities_db["geometry"].area/ 10**6 # in km
    # Get names in uppercase
    cities_db_with_population["NAME"] = cities_db_with_population["AccentCity"].map(lambda x: x.upper())

    # Get population
    cities_population = cities_db_with_population[["Population", "NAME"]]
    cities_population.dropna(inplace=True)
    # Find intersection
    intersection = pd.merge(cities_db, cities_population, how='inner', on=['NAME'])

    # Find density of population
    intersection["DENSITY_OF_POPULATION"] = intersection["Population"] / intersection["AREA"]

    # Find main statistic parameters
    population_statistics = basic_statistic_values(intersection["Population"])
    print("Min population:", population_statistics['min'])
    print("Max population:", population_statistics['max'])
    print("Mean population:", population_statistics['mean'])
    print("Median population:", population_statistics['median'])
    print("Std population:", population_statistics['std'])

    area_statistics = basic_statistic_values(intersection["AREA"])
    print("Min area:", area_statistics['min'])
    print("Max area:", area_statistics['max'])
    print("Mean area:", area_statistics['mean'])
    print("Median area:", area_statistics['median'])
    print("Std area:", area_statistics['std'])

    density_of_population_statistics = basic_statistic_values(intersection["DENSITY_OF_POPULATION"])
    print("Min density of population:", density_of_population_statistics['min'])
    print("Max density of population:", density_of_population_statistics['max'])
    print("Mean density of population:", density_of_population_statistics['mean'])
    print("Median density of population:", density_of_population_statistics['median'])
    print("Std density of population:", density_of_population_statistics['std'])

    # Create plots
    create_plot(intersection, "Population", "population", 30, "n cities", "Population")
    create_plot(intersection, "AREA", "area", 30, "n cities", "Area")
    create_plot(intersection, "DENSITY_OF_POPULATION", "density_of_population", 30, "n_cities", "Density of population")





