import geopandas as gpd
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

continents = ["Europe", "Asia", "Africa", "Australia", "North_America", "South_America"]

population_ranges = [100000, 500000, 1000000]

def get_country_codes():
    europe_iso_countries = pd.read_html("https://www.countrycallingcodes.com/iso-country-codes/europe-codes.php")[1][1]
    europe_iso_countries.dropna(inplace=True)
    europe_iso_countries = europe_iso_countries[europe_iso_countries.map(lambda x: len(str(x)) == 2)]
    asia_iso_countries = pd.read_html("https://www.countrycallingcodes.com/iso-country-codes/asia-codes.php")[1][1]
    asia_iso_countries.dropna(inplace=True)
    asia_iso_countries = asia_iso_countries[asia_iso_countries.map(lambda x: len(str(x)) == 2)]
    africa_iso_countries = pd.read_html("https://www.countrycallingcodes.com/iso-country-codes/africa-codes.php")[1][1]
    africa_iso_countries.dropna(inplace=True)
    africa_iso_countries = africa_iso_countries[africa_iso_countries.map(lambda x: len(str(x)) == 2)]
    australia_iso_countries = pd.read_html("https://www.countrycallingcodes.com/iso-country-codes/australia-codes.php")[1][1]
    australia_iso_countries.dropna(inplace=True)
    australia_iso_countries = australia_iso_countries[australia_iso_countries.map(lambda x: len(str(x)) == 2)]
    north_america_iso_countries = pd.read_html("https://www.countrycallingcodes.com/iso-country-codes/north-america-codes.php")[1][1]
    north_america_iso_countries.dropna(inplace=True)
    north_america_iso_countries = north_america_iso_countries[north_america_iso_countries.map(lambda x: len(str(x)) == 2)]
    south_america_iso_countries = pd.read_html("https://www.countrycallingcodes.com/iso-country-codes/south-america-codes.php")[1][1]
    south_america_iso_countries.dropna(inplace=True)
    south_america_iso_countries = south_america_iso_countries[south_america_iso_countries.map(lambda x: len(str(x)) == 2)]

    iso_codes = pd.DataFrame()
    iso_codes["Europe"] = europe_iso_countries
    iso_codes["Asia"] = asia_iso_countries
    iso_codes["Africa"] = africa_iso_countries
    iso_codes["Australia"] = australia_iso_countries
    iso_codes["North_America"] = north_america_iso_countries
    iso_codes["South_America"] = south_america_iso_countries
    return iso_codes


def basic_statistic_values(df):
    min = df.min()
    max = df.max()
    mean = df.mean()
    median = df.median()
    std = df.std()
    return {"min" : min, "max" : max, "mean" : mean, "median" : median,  "std" : std}


def create_plot(df, col, name, n_bins, y_tic, x_tic):
    title = col.lower().capitalize()
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
    ax1.set_title(title)
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
    plt.savefig(name + "_" + title + ".png", dpi=300)


if __name__ == "__main__":
    # Variable for storing results
    results = {}

    world_cities_db_with_area = gpd.read_file("cities.shp")
    world_cities_db_with_population = pd.read_csv("worldcitiespop.txt", encoding = "ISO-8859-1")

    # Get cities area
    world_cities_db_with_area = world_cities_db_with_area.to_crs({'init': 'epsg:3857'})
    world_cities_db_with_area["AREA"] = world_cities_db_with_area["geometry"].area/ 10**6 # in km

    # Get names in uppercase
    world_cities_db_with_population["NAME"] = world_cities_db_with_population["AccentCity"].map(lambda x: x.upper())
    world_cities_db_with_population = world_cities_db_with_population[["NAME", "Country", "Population"]]
    world_cities_db_with_population.dropna(inplace=True)

    # Get world cities db with density of population
    world_cities_db_with_density = pd.merge(world_cities_db_with_area, world_cities_db_with_population, how="inner", on=["NAME"])
    world_cities_db_with_density["Density"] = world_cities_db_with_density["Population"] / world_cities_db_with_density["AREA"]

    world_cities_db_with_area = pd.merge(world_cities_db_with_area, world_cities_db_with_population, how="inner", on=["NAME"])

    # Get basic statistics for world
    world_population_statistics = basic_statistic_values(world_cities_db_with_population["Population"])
    world_area_statistics = basic_statistic_values(world_cities_db_with_area["AREA"])
    world_density_statistics = basic_statistic_values(world_cities_db_with_density["Density"])
    results["World"] = {"population" : world_population_statistics, "area" : world_area_statistics, "density" : world_density_statistics}

    # Create hists for world
    create_plot(world_cities_db_with_population, "Population", "world", 30, "n cities", "Population")
    create_plot(world_cities_db_with_area, "AREA", "world", 30, "n cities", r'$\mathregular{km^2}$')
    create_plot(world_cities_db_with_density, "Density", "world", 30, "n cities", "Density")

    iso_codes = get_country_codes()

    # Make analysis for continents
    for continent in continents:
        # Get iso codes for specific continent
        continent_iso_codes = pd.DataFrame()
        continent_iso_codes["Country"] = iso_codes[continent].dropna().map(lambda x: x.lower())

        # Create db with population in specific continent and apply analysis
        continent_cities_with_population = pd.merge(world_cities_db_with_population, continent_iso_codes, how="inner", on=["Country"])
        continent_population_statistics = basic_statistic_values(continent_cities_with_population["Population"])

        # Create db with area in specific continent and apply analysis
        continent_cities_with_area = pd.merge(world_cities_db_with_area, continent_iso_codes, how="inner", on=["Country"])
        continent_area_statistics = basic_statistic_values(continent_cities_with_area["AREA"])

        # Create db with dencity in specific continent and apply analysis
        continent_cities_with_density = pd.merge(world_cities_db_with_density, continent_iso_codes, how="inner", on=["Country"])
        continent_density_statistics = basic_statistic_values(continent_cities_with_density["Density"])

        results[continent] = {"population" : continent_population_statistics, "area" : continent_area_statistics, "density" : continent_density_statistics}

        # Create plot for continents
        create_plot(continent_cities_with_population, "Population", continent, 30, "n cities", "Population")
        create_plot(continent_cities_with_area, "AREA", continent, 30, "n cities", r'$\mathregular{km^2}$')
        create_plot(continent_cities_with_density, "Density", continent, 30, "n cities", "Density")

    # Make analysis for cities with population larger than 100 000
    for number in population_ranges:
        world_cities_db_with_population_filter = world_cities_db_with_population[world_cities_db_with_population["Population"] > number]
        world_cities_db_with_population_filter_statistics = basic_statistic_values(world_cities_db_with_population_filter["Population"])

        world_cities_db_with_area_filter = world_cities_db_with_density[world_cities_db_with_density["Population"] > number]
        world_cities_db_with_area_filter_statistics = basic_statistic_values(world_cities_db_with_area_filter["AREA"])
        world_cities_db_with_density_filter_statistics = basic_statistic_values(world_cities_db_with_area_filter["Density"])

        results["larger_than_" + str(number)] = {"population" : world_cities_db_with_population_filter_statistics, "area" : world_cities_db_with_area_filter_statistics,
                                         "density" : world_cities_db_with_density_filter_statistics}

        # Create plot for continents
        create_plot(world_cities_db_with_population_filter, "Population", "population_larger_" + str(number), 30, "n cities", "Population")
        create_plot(world_cities_db_with_area_filter, "AREA", "population_larger_" + str(number), 30, "n cities", r'$\mathregular{km^2}$')
        create_plot(world_cities_db_with_area_filter, "Density", "population_larger_" + str(number), 30, "n cities", "Density")


    results_df = pd.DataFrame(results)

    # results_df.to_json("results.json")














