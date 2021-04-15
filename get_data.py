import pandas as pd 
from pandas import DataFrame 
from datetime import datetime, timedelta
import tqdm
import numpy as np
import pygsheets
import fbprophet


# names of vacccines 
NAME_PFIZER = "pfizer"
NAME_PFIZER_EXTRA = f"{NAME_PFIZER}_extra"
NAME_MODERNA = "moderna"
NAME_MODERNA_EXTRA = f"{NAME_MODERNA}_extra"
NAME_JJ = "johnson_johnson"
NAME_CUREVAC = "curevac"
NAME_SANOFI = "sanofi"
NAME_ASTRA = "astra"

# different vaccinations strategies, 
# scenario_a = current STIKO vaccination strategy, scenario_b = 
VACCINATIONS_STRATEGIES = {"scenario_a": {NAME_PFIZER: 6,
                                          NAME_MODERNA: 6, 
                                          NAME_SANOFI: 6, 
                                          NAME_CUREVAC: 6, 
                                          NAME_ASTRA: 12,
                                          NAME_JJ: -1}, 
                                   
                         "scenario_b": {NAME_PFIZER: 12,
                                        NAME_MODERNA: 12, 
                                        NAME_SANOFI: 12, 
                                        NAME_CUREVAC: 12, 
                                        NAME_ASTRA: 12,
                                        NAME_JJ: -1}
                         } 
                        
# relevant urls RKI   
URL_RKI_DELIVERIES_LATEST = "https://impfdashboard.de/static/data/germany_deliveries_timeseries_v2.tsv"
URL_RKI_IMPFREIHE_LATEST = "https://impfdashboard.de/static/data/germany_vaccinations_timeseries_v2.tsv"

# destatis data 
# source https://de.statista.com/statistik/daten/studie/1200499/umfrage/prognose-zu-lieferungen-von-corona-impfstoffen/#professional

q4_2020 = {"quarter": [4], "year": [2020], "contract_name": [NAME_PFIZER], "amount": [1.3], "amount_usable": [1.3]}

q1_2021 = {"quarter": 1, "year": 2021, 
           "contract_name": [NAME_PFIZER, NAME_MODERNA, NAME_ASTRA], 
           "amount": [10.9, 1.8, 5.6], 
           "amount_usable": [10.9, 1.8, 2.9]}
# due to the decision to only vaccinate citiziens 60+, all persons vaccinated with AstraZeneca now have to receive a 
# another mRNA vaccine, the usable doses are reduced by equal amounts 
AMOUNT_ASTRA_FOR_SECOND_VACC = 2.6

q2_2021 = {"quarter": 2, "year": 2021,
           "contract_name": [NAME_MODERNA, NAME_CUREVAC, NAME_PFIZER_EXTRA, NAME_JJ, NAME_PFIZER, NAME_ASTRA], 
           "amount": [6.4, 3.5, 8.7, 10.1, 31.5, 16.9], 
           "amount_usable": [6.4-AMOUNT_ASTRA_FOR_SECOND_VACC/5, 
                             3.5-AMOUNT_ASTRA_FOR_SECOND_VACC/5, 
                             8.7-AMOUNT_ASTRA_FOR_SECOND_VACC/5, 
                             10.1-AMOUNT_ASTRA_FOR_SECOND_VACC/5, 
                             31.5 - AMOUNT_ASTRA_FOR_SECOND_VACC/5, 
                             16.9]}
q3_2021 = {"quarter": 3, "year": 2021,
           "contract_name": [NAME_MODERNA, NAME_MODERNA_EXTRA, NAME_CUREVAC, NAME_PFIZER_EXTRA, NAME_JJ, NAME_PFIZER, NAME_ASTRA], 
           "amount": [17.6, 9.1, 9.4, 17.1, 22, 17.6, 33.8], 
           "amount_usable": [17.6, 9.1, 9.4, 17.1, 22, 17.6, 33.8]}

q4_2021 = {"quarter": 4,  "year": 2021,
           "contract_name": [NAME_SANOFI, NAME_MODERNA, NAME_MODERNA_EXTRA, NAME_CUREVAC, NAME_JJ, NAME_MODERNA,  NAME_PFIZER_EXTRA], 
           "amount": [27.5, 24.6, 18.3, 11.7, 10.8, 4.6, 2.7], 
           "amount_usable": [27.5, 24.6, 18.3, 11.7, 10.8, 4.6, 2.7]}

QUARTERLY_DELIVERIES_STATISTA = [q4_2020, q1_2021, q2_2021, q3_2021, q4_2021]

PATH_CUSTOM_DISTRIBUTION_QUARTERLY_DELIVERIES = "impftracker/data/custom_time_distribution_vacc.xlsx"
SHEET_SCENARIO_JJ = "custom_jj"

# helper type of vaccinations necessary for tableau plotting
TYPE_VACCS_HELPER = {"dosen_erst_minus_zweit_kum": "helper_1st", 
                     "dosen_zweit_kum": "helper_2nd", 
                     "dosen_reserve_kum": "helper_reserve"}


def prep_rki_deliveries(path_rki: str = URL_RKI_DELIVERIES_LATEST):   
    df_rki = pd.read_csv(path_rki,
                         sep="\t",
                         names=["date", "impfstoff", "region", "amount"], 
                         header=0, 
                         parse_dates=[0])

    df_rki["type_vacc"] = df_rki["impfstoff"].map({"comirnaty": NAME_PFIZER, "astra": NAME_ASTRA, " moderna": NAME_MODERNA})
    #df_rki["state_code"] = df_rki["region"].str.split("-", expand=True)[1]
    df_rki["year"] = df_rki["date"].dt.year
    df_rki["week_of_year"] = df_rki["date"].dt.isocalendar().week
    df_rki["quarter"] = df_rki["date"].dt.quarter
    df_rki["future"] = False
    df_rki["first_day_of_week"] = df_rki['date'].apply(lambda x: (x - timedelta(days=x.dayofweek)))

    df_rki_weekly = (df_rki.groupby(["year", "week_of_year", "type_vacc"], as_index=False)
                     .agg(**{**{col: (col, "first") for col in df_rki.columns},
                             **{"amount_type_vacc_de_rki_weekly": ("amount", "sum")}})
                     .drop(columns=["date", "region"])
                    )
    df_rki_weekly["source"] = "impfdashboard"
    df_rki_weekly["percent_of_weekly_amount"] = df_rki_weekly.groupby(["year", "week_of_year"])["amount"].transform(lambda x: x/x.sum())
    return df_rki_weekly 


def prep_rki_vaccs(path: str = URL_RKI_IMPFREIHE_LATEST, cols_impfreihe: list = None, agg_week: bool = True):
    df_ir_de = pd.read_csv(path, sep="\t", parse_dates=[0])
    df_ir_de["first_day_of_week"] = df_ir_de['date'].apply(lambda x: (x - timedelta(days=x.dayofweek)))
    if cols_impfreihe is None: 
        cols_impfreihe = df_ir_de.columns
    
    cols_numeric = df_ir_de.select_dtypes(include=[np.number]).columns
    cols_non_numeric = df_ir_de.columns.difference(cols_numeric)
    cols_sum = {col: "sum" for col in cols_numeric}
    cols_max = {col: "max" for col in cols_numeric if ("kumulativ" in col or "indikation" in col)}
    cols_avg = {col: "mean" for col in cols_numeric if "quote" in col}
    cols_first = {col: "first" for col in cols_non_numeric}

    agg = {**cols_sum, **cols_max, **cols_first, **cols_avg}
    if agg_week: 
        return df_ir_de.groupby(["first_day_of_week"]).agg(agg).reset_index(drop=True)
    else: 
        return df_ir_de


def read_custom_vacc_distribution_time(path: str = PATH_CUSTOM_DISTRIBUTION_QUARTERLY_DELIVERIES,
                                       sheet_name: str = SHEET_SCENARIO_JJ):
    
    df_distr_vacc_melted =  pd.melt(pd.read_excel(path, sheet_name=sheet_name), 
                                    id_vars=["year", "quarter", "week_of_year", "first_day_of_week"], 
                                    var_name="type_vacc",
                                    value_name="distr_vacc_custom")
    # sanity check
    df_sanity_check = df_distr_vacc_melted.dropna(subset=["distr_vacc_custom"]).groupby(["year", "quarter", "type_vacc"])[["distr_vacc_custom"]].sum()
    assert np.allclose(df_sanity_check["distr_vacc_custom"], 1.0)
    
    return df_distr_vacc_melted


def disaggregate_destatis_weekly(df_destatis: DataFrame, 
                                 path_custom_distr_deliv: str = PATH_CUSTOM_DISTRIBUTION_QUARTERLY_DELIVERIES,
                                 scenario_distribution: str = SHEET_SCENARIO_JJ
                                 ): 
    weekstarts = (pd.Series(pd.date_range(start="2020-12-21", end="2021-12-31"))
                  .apply(lambda x: (x - timedelta(days=x.dayofweek))).unique())

    df_destatis_weekly = pd.DataFrame({"first_day_of_week": weekstarts})
    
    df_destatis_weekly["year"] = df_destatis_weekly["first_day_of_week"].dt.year
    df_destatis_weekly["week_of_year"] = df_destatis_weekly["first_day_of_week"].dt.isocalendar().week
    df_destatis_weekly["quarter"] = df_destatis_weekly["first_day_of_week"].dt.quarter
    
    df_destatis_weekly["distr_factor_equal"] = (df_destatis_weekly
                                                .groupby(["year", "quarter"])
                                                ["week_of_year"]
                                                .transform(lambda x: 1/len(x))
                                                )
    
    df_destatis_weekly = df_destatis_weekly.merge((df_destatis
                                               .set_index(["year", "quarter"])
                                               [["amount_type_vacc", "type_vacc", "amount_type_vacc_usable"]]
                                               .rename(columns={"amount_type_vacc": "amount_type_vacc_de_quarter", 
                                                                "amount_type_vacc_usable": "amount_type_vacc_de_quarter_usable"})
                                              ), 
                                              left_on=["year", "quarter"], right_index=True, 
                                              how="left")
    if path_custom_distr_deliv is None: 
        df_destatis_weekly["distr_factor"] = df_destatis_weekly["distr_factor_equal"]
    else: 
        df_distr_factors_weekly = read_custom_vacc_distribution_time(path_custom_distr_deliv, sheet_name=scenario_distribution)
        df_destatis_weekly = df_destatis_weekly.merge(df_distr_factors_weekly.set_index(["year", "quarter", "week_of_year", "type_vacc"])[["distr_vacc_custom"]], 
                                                      how="left",
                                                      validate="m:1",
                                                      left_on=["year", "quarter", "week_of_year", "type_vacc"], 
                                                      right_index=True)
        df_destatis_weekly["distr_factor"] = np.where(df_destatis_weekly["distr_vacc_custom"].isna(), 
                                                      df_destatis_weekly["distr_factor_equal"], 
                                                      df_destatis_weekly["distr_vacc_custom"])
    
    df_destatis_weekly["amount_type_vacc_de_ds_weekly"] = df_destatis_weekly["amount_type_vacc_de_quarter"] * df_destatis_weekly["distr_factor"]
    df_destatis_weekly["amount_type_vacc_de_ds_weekly_usable"] = df_destatis_weekly["amount_type_vacc_de_quarter_usable"] * df_destatis_weekly["distr_factor"]
    df_destatis_weekly["source"] = "destatis"
    df_destatis_weekly["scenario_distribution"] = scenario_distribution
    return df_destatis_weekly


def prep_destatis(data: list = QUARTERLY_DELIVERIES_STATISTA):   
    df_destatis = pd.concat([pd.DataFrame(q) for q in data], ignore_index=True)
    df_destatis["amount_contract"] = df_destatis["amount"] * 10**6
    df_destatis["amount_usable"] = df_destatis["amount_usable"] * 10**6

    df_destatis["type_vacc"] = df_destatis["contract_name"].str.replace("_extra", "")
    df_destatis["amount_type_vacc"] = (df_destatis.groupby(["year", "quarter", "type_vacc"])
                                       ["amount_contract"]
                                       .transform(lambda x: x.sum())
                                       )
    df_destatis["amount_type_vacc_usable"] = (df_destatis.groupby(["year", "quarter", "type_vacc"])
                                             ["amount_usable"]
                                             .transform(lambda x: x.sum())
                                             )
    df_destatis = (df_destatis
                   .groupby(["year", "quarter", "type_vacc"], as_index=False)
                   .agg(**{**{col: (col, "first") for col in df_destatis},
                           **{"amount_contract": ("amount_contract", "sum")}})
                   .reset_index(drop=False)
                   )
    
    df_destatis_weekly = disaggregate_destatis_weekly(df_destatis=df_destatis)
    
    return df_destatis_weekly

    
def combine_datasources_deliveries(df_rki_deliv_weekly: DataFrame = None , 
                                   df_destatis_weekly: DataFrame = None): 
    
    if df_rki_deliv_weekly is None: 
        df_rki_weekly_type_vacc = prep_rki_deliveries()
    if df_destatis_weekly is None: 
        df_destatis_weekly = prep_destatis()
        
    cols_merge_rki = ["first_day_of_week", "type_vacc", "amount_type_vacc_de_rki_weekly"]
    
    cols_merge_destatis = ["amount_type_vacc_de_ds_weekly", "amount_type_vacc_de_ds_weekly_usable"]

    mask_latest_common_date = df_destatis_weekly["first_day_of_week"] <= df_rki_weekly_type_vacc.first_day_of_week.max()

    df_de_weekly_type_vacc = pd.merge(df_rki_weekly_type_vacc[cols_merge_rki],
                            df_destatis_weekly[mask_latest_common_date].set_index(["first_day_of_week", "type_vacc"])[cols_merge_destatis],
                            how="outer", 
                            left_on=["first_day_of_week", "type_vacc"], 
                            right_index=True)

    cols_merge_destatis = ["first_day_of_week", "type_vacc", 
                           "amount_type_vacc_de_ds_weekly", 
                           "amount_type_vacc_de_ds_weekly_usable"]
    df_de_weekly_type_vacc = pd.concat([df_de_weekly_type_vacc,
                              df_destatis_weekly[~mask_latest_common_date][cols_merge_destatis]], 
                              axis=0, ignore_index=True)
    
    df_de_weekly_type_vacc["year"] = df_de_weekly_type_vacc["first_day_of_week"].dt.year
    df_de_weekly_type_vacc["week_of_year"] = df_de_weekly_type_vacc["first_day_of_week"].dt.isocalendar().week
    df_de_weekly_type_vacc["quarter"] = df_de_weekly_type_vacc["first_day_of_week"].dt.quarter

    mask_is_nan = df_de_weekly_type_vacc["amount_type_vacc_de_rki_weekly"].isna()
    df_de_weekly_type_vacc["amount_type_vacc_de_weekly"] = np.where(~mask_is_nan,
                                                                    df_de_weekly_type_vacc["amount_type_vacc_de_rki_weekly"], 
                                                                    df_de_weekly_type_vacc["amount_type_vacc_de_ds_weekly"])
    df_de_weekly_type_vacc["amount_type_vacc_de_weekly_usable"] = np.where(~mask_is_nan,
                                                                     df_de_weekly_type_vacc["amount_type_vacc_de_rki_weekly"], 
                                                                     df_de_weekly_type_vacc["amount_type_vacc_de_ds_weekly_usable"])
    mask_latest_common_date = df_destatis_weekly["first_day_of_week"] <= df_rki_weekly_type_vacc.first_day_of_week.max()
    df_de_weekly_type_vacc["future"] =  np.where(mask_latest_common_date, False, True) 
    
    return df_de_weekly_type_vacc


def combine_rki_datasources(df_de_weekly_type_vacc): 
    
    df_weekly_deliv_type_vacc = prep_rki_deliveries()
    df_weekly_vac = prep_rki_vaccs().set_index("first_day_of_week")

    df_weekly_deliv = (df_weekly_deliv_type_vacc
                       .groupby("first_day_of_week")
                       .agg(**{"amount_vacc_week_distributed": ("amount_type_vacc_de_rki_weekly", "sum")})
                       )
    df_reindex = df_de_weekly_type_vacc.groupby("first_day_of_week")[["type_vacc"]].first()
    df_rki_weekly = (df_weekly_vac.reindex(index=df_reindex.index)
                     .merge(df_weekly_deliv, on="first_day_of_week", validate="1:1", how="left")
                     )
    return df_rki_weekly.reset_index(drop=False)


def _compute_number_of_vaccinations_by_type(df_de_type_vacc: DataFrame,
                                            type_vacc: str,
                                            scenario: dict,
                                            suffix_scenario: str,
                                            col_amount_vacc: str ="amount_type_vacc_de_weekly_usable"): 
    
    period = scenario[type_vacc]
    df_de_type_vacc.sort_values(by="first_day_of_week", inplace=True)
    df_de_type_vacc.reset_index(drop=True, inplace=True)

    col_amount_for_first_vacc = f"amount_available_for_first_vacc_{suffix_scenario}"
    col_n_first_vacc = f"n_first_vacc_{suffix_scenario}"
    col_n_second_vacc = f"n_second_vacc_{suffix_scenario}"
    col_n_fully_immune_cum = f"n_fully_immune_{suffix_scenario}"
    
    if period == -1: 
        df_de_type_vacc[col_amount_for_first_vacc] = df_de_type_vacc[col_amount_vacc].to_numpy()
        df_de_type_vacc[col_n_first_vacc] = df_de_type_vacc[col_amount_vacc]
        df_de_type_vacc[col_n_second_vacc] = 0
        df_de_type_vacc[col_n_fully_immune_cum] = df_de_type_vacc[col_n_first_vacc].cumsum()

    else: 
        init_amount = df_de_type_vacc[col_amount_vacc].to_numpy()

        n_first_vacc = np.minimum(init_amount[0:period], 
                                  init_amount[0:period]/2 + init_amount[period: period*2]/2)

        amount_available_for_first_vac = []
        
        for i, amount in enumerate(df_de_type_vacc[col_amount_vacc]): 
            if i < period: 
                amount_available_for_first_vac += [n_first_vacc[i]]
            else: 
                amount_available_for_first_vac += [max(0, init_amount[i] - amount_available_for_first_vac[i-period])]

        df_de_type_vacc[col_amount_for_first_vacc] = amount_available_for_first_vac
        df_de_type_vacc[col_n_first_vacc] = np.minimum(df_de_type_vacc[col_amount_for_first_vacc], 
                                                   df_de_type_vacc[col_amount_vacc].shift(-period).fillna(0.0)/2 + df_de_type_vacc[col_amount_for_first_vacc]/2)
        df_de_type_vacc[col_n_first_vacc].iloc[0:period] = df_de_type_vacc[col_amount_for_first_vacc].iloc[0:period]
        
        df_de_type_vacc[col_n_second_vacc] = df_de_type_vacc[col_n_first_vacc].shift(period).fillna(0.0)
        df_de_type_vacc[col_n_fully_immune_cum] = df_de_type_vacc[col_n_second_vacc].cumsum()
    return df_de_type_vacc


def compute_number_of_vaccinations(df_de_weekly: DataFrame, scenario: dict, suffix_scenario: str): 

    list_df_type_vacs = []
    for type_vacc, df_de_type_vacc in df_de_weekly.groupby("type_vacc"): 
            list_df_type_vacs += [_compute_number_of_vaccinations_by_type(df_de_type_vacc, 
                                                                          type_vacc=type_vacc, 
                                                                          scenario=scenario, 
                                                                          suffix_scenario=suffix_scenario
                                                                          )]
    return pd.concat(list_df_type_vacs, ignore_index=True)


def predict_vaccinations(prediction_horizon_weeks: int = 4): 
    df_rki_daily = prep_rki_vaccs(agg_week=False)
    
    df_1st = (df_rki_daily
              [["date", "dosen_erst_differenz_zum_vortag"]]
              .rename(columns={"date":"ds", "dosen_erst_differenz_zum_vortag": "y"})
              .copy()
              )
    df_2nd = (df_rki_daily
              [["date", "dosen_zweit_differenz_zum_vortag"]]
              .rename(columns={"date":"ds", "dosen_zweit_differenz_zum_vortag": "y"})
              .copy()
              )
    
    model_1st = fbprophet.Prophet(yearly_seasonality=False, daily_seasonality=True)
    model_1st.fit(df_1st)
    
    model_2nd = fbprophet.Prophet(yearly_seasonality=False, daily_seasonality=True)
    model_2nd.fit(df_2nd)
    date_first_prediction = df_rki_daily["date"].max() + pd.Timedelta(days=1)

    df_future = pd.DataFrame([pd.to_datetime(date_first_prediction) + pd.Timedelta(days=i) for i in range(prediction_horizon_weeks*7)], 
                             columns=["ds"])
    
    df_future["ds"] = df_future["ds"].dt.date

    df_predict_1st = model_1st.predict(df_future).set_index("ds")
    df_predict_2nd = model_2nd.predict(df_future).set_index("ds")

    df_1st = (pd.concat([df_1st.set_index("ds")["y"], df_predict_1st["yhat"]])
              .reset_index(drop=False)
              .rename(columns={0: "dosen_erst_mit_projektion"})
              )
    df_2nd = (pd.concat([df_2nd.set_index("ds")["y"], df_predict_2nd["yhat"]])
              .reset_index(drop=False)
              .rename(columns={0: "dosen_zweit_mit_projektion"})
              .drop(columns=["ds"])
              )
    
    df_vacc_with_predict = pd.concat([df_1st, df_2nd], axis=1)
    df_vacc_with_predict["first_day_of_week"] = df_vacc_with_predict['ds'].apply(lambda x: (x - timedelta(days=x.dayofweek)))
    return df_vacc_with_predict
    
    
def cumsum_vaccinations(df_rki_weekly: DataFrame, 
                        df_vacc_predicted: DataFrame, 
                        last_rki_vacc_update: datetime.date, 
                        last_rki_deliv_update: datetime.date,
                        ):
    mask_stop_cumsum = df_rki_weekly["first_day_of_week"] > last_rki_deliv_update    
    
    doses_reserve =(df_rki_weekly["amount_vacc_week_distributed"] 
                    - df_rki_weekly["dosen_erst_differenz_zum_vortag"] 
                    - df_rki_weekly["dosen_zweit_differenz_zum_vortag"])

    df_rki_weekly["dosen_reserve"] = np.where(mask_stop_cumsum, np.nan, doses_reserve)
    
    agg = {"dosen_erst_mit_projektion": ("dosen_erst_mit_projektion", "sum"), 
           "n_days_in_week": ("dosen_erst_mit_projektion", len),
           "dosen_zweit_mit_projektion": ("dosen_zweit_mit_projektion", "sum")}
    df_vacc_pred_agg = df_vacc_predicted.groupby("first_day_of_week").agg(**agg)
    
    df_rki_weekly = df_rki_weekly.merge(df_vacc_pred_agg, how="left", validate="1:1", on="first_day_of_week")
                                              
    mask_stop_cumsum = (df_rki_weekly["first_day_of_week" ] > last_rki_vacc_update) & (df_rki_weekly["n_days_in_week"] != 7.0)
    df_rki_weekly["dosen_erst_kum"] = np.where(mask_stop_cumsum, np.nan, df_rki_weekly["dosen_erst_mit_projektion"].cumsum())
    df_rki_weekly["dosen_zweit_kum"] = np.where(mask_stop_cumsum, np.nan, df_rki_weekly["dosen_zweit_mit_projektion"].cumsum())
    df_rki_weekly["dosen_erst_minus_zweit_kum"] = np.maximum(0, df_rki_weekly["dosen_erst_kum"] -  df_rki_weekly["dosen_zweit_kum"]) 
    df_rki_weekly["dosen_reserve_kum"] = - df_rki_weekly["dosen_reserve"].cumsum()
    
    return df_rki_weekly


def save_to_googlesheets(df_save: DataFrame): 
    #authorization
    google_client = pygsheets.authorize(service_file='archive/data/impf-dashboard-064b3ec57a70.json')
    sh = google_client.open('impftracker')
    #select the first sheet 
    print("clearing")
    wks = sh[0]
    wks.clear()
    print("saving")
    wks.set_dataframe(df_save.sort_values(by="first_day_of_week"), (1,1))


def prep_data_for_tableau(df_de_weekly_type_vacc: DataFrame, 
                          df_rki_weekly_vacc: DataFrame,
                          last_rki_vacc_update: datetime.date, 
                          type_vaccs_helper: dict = TYPE_VACCS_HELPER, 
                          cols_relevant: list = None,
                          ): 
    
    cols_melt = type_vaccs_helper.keys()
    df_type_vacc_helper = (pd.melt(df_rki_weekly_vacc.set_index("first_day_of_week")[cols_melt], 
                                   value_name="amount_actual", var_name="col_rki_ir", ignore_index=False)
                           .reset_index(drop=False)
                           )
    df_type_vacc_helper["type_vacc"] = df_type_vacc_helper["col_rki_ir"].map(type_vaccs_helper)
    df_type_vacc_helper["year"] = df_type_vacc_helper["first_day_of_week"].dt.year
    df_type_vacc_helper["week_of_year"] = df_type_vacc_helper["first_day_of_week"].dt.isocalendar().week
    df_type_vacc_helper["quarter"] = df_type_vacc_helper["first_day_of_week"].dt.quarter
    #df_type_vacc_helper["country"] = "de"
    df_type_vacc_helper["future"] = df_type_vacc_helper["first_day_of_week"] > last_rki_vacc_update
    if cols_relevant is None: cols_relevant = df_de_weekly_type_vacc.columns
    df_de_weekly = pd.concat([df_de_weekly_type_vacc[cols_relevant], 
                              df_type_vacc_helper.drop(columns=["col_rki_ir"])], 
                             axis=0, ignore_index=True)
    df_de_weekly.drop(columns=["amount_type_vacc_de_ds_weekly", 
                               "amount_type_vacc_de_ds_weekly_usable", 
                               "amount_type_vacc_de_rki_weekly"], inplace=True)
    return df_de_weekly.sort_values(by="first_day_of_week")


def compute_reference_line(df_rki_weekly_vacc: DataFrame, 
                           df_de_weekly_type_vacc: DataFrame,
                           date_start: str = "2021-04-01", 
                           date_end: str = "2021-07-31", 
                           num_people_fully_immune = 52_500_000):
     
    df_rl = pd.DataFrame({"date": pd.date_range(start=date_start, end=date_end, freq="d")})

    df_rl["first_day_of_week"] =  df_rl['date'].apply(lambda x: (x - timedelta(days=x.dayofweek)))

    num_people_fully_immune_start = (df_rki_weekly_vacc
                                     .set_index("first_day_of_week")
                                     .sort_index()
                                     .truncate(before=date_start)
                                     ["personen_voll_kumulativ"]
                                     .iloc[0])
    num_people_fully_immune = 52_500_000

    n_second_per_day = (num_people_fully_immune - num_people_fully_immune_start)/df_rl.shape[0]

    df_rl["amount_2nd_reference_line"] = n_second_per_day
    df_rl["amount_2nd_reference_line"] = df_rl["amount_2nd_reference_line"].cumsum() + num_people_fully_immune_start
    df_rl_agg = df_rl.groupby("first_day_of_week")[["amount_2nd_reference_line"]].max()
    df_de_weekly = df_de_weekly_type_vacc.merge(df_rl_agg, 
                                                how="left", right_index=True, left_on="first_day_of_week", 
                                                validate="m:1")
    return df_de_weekly 


def main(): 

    
    # combine distributed vaccinations, number of vaccinations, and statista projection into single dataFrame 
    df_de_weekly_type_vacc = combine_datasources_deliveries()
    
    df_rki_weekly_vacc = combine_rki_datasources(df_de_weekly_type_vacc=df_de_weekly_type_vacc)
    last_rki_vacc_update = df_rki_weekly_vacc.dropna(subset=["dosen_kumulativ"])["first_day_of_week"].max()
    last_rki_deliv_update = df_rki_weekly_vacc.dropna(subset=["amount_vacc_week_distributed"])["first_day_of_week"].max()
    
    for suffix_scenario, scenario in VACCINATIONS_STRATEGIES.items(): 
        df_de_weekly_type_vacc = compute_number_of_vaccinations(df_de_weekly_type_vacc, 
                                                                scenario=scenario,
                                                                suffix_scenario=suffix_scenario)

    # predictions of future vaccinations 
    df_vacc_predicted = predict_vaccinations()
    
    df_rki_weekly_vacc = cumsum_vaccinations(df_rki_weekly_vacc,
                                             df_vacc_predicted=df_vacc_predicted,
                                             last_rki_deliv_update=last_rki_deliv_update, 
                                             last_rki_vacc_update=last_rki_vacc_update)
    #return df_rki_weekly_vacc
    df_de_weekly_type_vacc = compute_reference_line(df_rki_weekly_vacc=df_rki_weekly_vacc, 
                                                    df_de_weekly_type_vacc=df_de_weekly_type_vacc)
    # prep data for Tableau 
    df_tableau = prep_data_for_tableau(df_de_weekly_type_vacc=df_de_weekly_type_vacc,
                                       df_rki_weekly_vacc=df_rki_weekly_vacc, 
                                       last_rki_vacc_update=last_rki_deliv_update)
    #return df_tableau
    save_to_googlesheets(df_tableau)
    return df_tableau
    

if __name__ == "main": 
    main()