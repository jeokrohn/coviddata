import logging
from github import Github, ContentFile
import pandas as pd
import numpy as np
import glob
import os
import zipfile

GITHUB_TOKEN = os.getenv('GITHUB_PUBLIC_READ_TOKEN')
assert GITHUB_TOKEN is not None

CSV_PATH = 'CSV'
GITHUB_REPO = 'CSSEGISandData/COVID-19'

log = logging.getLogger(__name__)


def get_csvs():
    g = Github(login_or_token=GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)
    c = repo.get_contents('csse_covid_19_data/csse_covid_19_daily_reports')

    local_csvs = glob.glob(os.path.join(CSV_PATH, '*.csv'))
    local_csvs = list(map(os.path.basename, local_csvs))
    need_dowload = [f for f in c if f.path.endswith('.csv') and os.path.basename(f.path) not in local_csvs]
    for cf in need_dowload:
        fn = os.path.basename(cf.path)
        log.info(f'writing {fn}')
        with open(os.path.join(CSV_PATH, fn), 'wb') as f:
            f.write(cf.decoded_content)


def path_to_date(path):
    path = os.path.basename(path)
    path = os.path.splitext(path)[0]
    date = pd.Timestamp(f'{path[-4:]}-{path[:2]}-{path[3:5]}')
    return date


def covid_df_from_csv(path):
    column_info = {
        'FIPS': '',
        'Admin2': '',
        'Province_State': 'Province/State',
        'Country_Region': 'Country/Region',
        'Last_Update': 'Last Update',
        'Lat': 'Latitude',
        'Long_': 'Longitude',
        'Confirmed': 'Confirmed',
        'Deaths': 'Deaths',
        'Recovered': 'Recovered',
        'Active': '',
        'Combined_Key': '',
    }

    def combined_key(row):
        keys = ['Admin2', 'Province_State', 'Country_Region']
        values = [row[key] for key in keys]
        r = ', '.join((v for v in values if v and pd.notna(v)))
        return r

    log.debug(f'reading {path}')
    df = pd.read_csv(path)
    df['Date'] = path_to_date(path)
    column_names = list(df.columns)
    for column_name, old_column_name in column_info.items():
        if column_name in column_names:
            continue
        if old_column_name:
            if old_column_name in column_names:
                log.debug(f'{path}: renaming {old_column_name} to {column_name}')
                df.rename(columns={old_column_name: column_name}, inplace=True)
                continue
            else:
                log.debug(f'{path}: adding missing column {column_name}')
                df[column_name] = np.nan
                continue
        if column_name in ['FIPS', 'Admin2']:
            log.debug(f'{path}: adding missing column {column_name}')
            df[column_name] = np.nan
        elif column_name == 'Active':
            log.debug(f'{path}: adding missing column {column_name}')
            df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
        elif column_name == 'Combined_Key':
            log.debug(f'{path}: adding missing column {column_name}')
            df['Combined_Key'] = df.apply(combined_key, axis=1)
        else:
            raise NotImplementedError(f'Unexpected column name: {column_name}')

    return df


def covid_preprocess(covid):

    # 'China' was reported as 'Mainland Chind' at some point
    # Looks like the 'Mainland China' records which overlap with the 'China' records don't have numbers for
    # Confirmed, Deaths, Recovered, or Active. We might as well get rid of them and then rename 'Mainland China' to
    # 'China'
    first_china = covid[covid.Country_Region == 'China'].Date.min()

    covid.drop(covid[(covid.Country_Region == 'Mainland China') & (covid.Date >= first_china)].index, inplace=True)
    covid.loc[covid.Country_Region == 'Mainland China', 'Country_Region'] = 'China'

    # set defaults for Deaths and Recovered
    for c in ['Deaths', 'Recovered']:
        covid[c].fillna(value=0, inplace=True)

    # set defaults for Deaths, Recovered, and Confirmed
    for c in ['Deaths', 'Recovered', 'Confirmed']:
        covid[c].fillna(value=0, inplace=True)
    covid.Active = covid.Confirmed - covid.Deaths - covid.Recovered
    return covid


def read_population():
    zip_files = glob.glob('API_SP*.zip')
    assert len(zip_files) == 1
    with zipfile.ZipFile(file=zip_files[0]) as zip_file:
        names = zip_file.namelist()
        data_file_name = next((n for n in names if n.startswith('API_SP')), None)
        assert data_file_name is not None
        with zip_file.open(data_file_name) as data_file:
            df = pd.read_csv(data_file, header=2)

    # drop empty columns
    df = df.dropna(axis='columns', how='all')

    # get value from last population column w/ non NaN value
    df['population'] = df.loc[:, '1960':].copy().ffill(axis=1).iloc[:, -1]

    # remove rows where we don't have a population value
    df = df[df.population.notnull()]

    # we only really need two columns
    df = df.loc[:, ['Country Name', 'population']]
    df = df.set_index('Country Name')

    return df


def read_population_un():
    """
    read population data obtained from https://population.un.org/wpp/Download/Standard/CSV/
    :return:
    """
    df = pd.read_csv('WPP2019_TotalPopulationBySex.csv', header=0)
    df = df[df.Time == 2019]
    df = df.set_index('Location')
    df['population'] = df.PopTotal * 1000
    return df


def top_impacted_countries(covid, n=10):
    # get sums per country and date
    country_data = covid.groupby(by=['Country_Region', 'Date']).Confirmed.sum()
    # only keep maximum per country
    max_confirmed_per_country = country_data.groupby(by='Country_Region').max()
    # .. and we only want the top 10 countries
    top_impacted = max_confirmed_per_country.nlargest(n=n, keep='all').to_frame()

    return top_impacted


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('github').setLevel(logging.INFO)
    get_csvs()
    population = read_population_un()

    csv_files = glob.glob(os.path.join(CSV_PATH, '*.csv'))
    csv_files.sort(reverse=True)
    covid = covid_preprocess(pd.concat(covid_df_from_csv(csv_file) for csv_file in csv_files))
    top_impacted_countries(covid=covid)

    # Need to find the country names in the covid data name which do not exist in the population dataset
    countries_in_covid = covid.Country_Region.unique()
    countries_in_population = list(population.index.values)
    covid_countries_not_in_population = [c for c in countries_in_covid if c not in countries_in_population]
    germany = covid[covid['Country_Region'] == 'Germany']

    # transform country names in population
    country_name_transforms = [('United States of America', 'US'),
                               ('Bolivia (Plurinational State of)', 'Bolivia'),
                               ('Brunei Darussalam', 'Brunei')]


if __name__ == '__main__':
    main()
