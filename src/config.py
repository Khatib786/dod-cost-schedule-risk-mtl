
# Column mappings for the DoD FPDS-NG dataset.

# Supports variations in header names across exports and years.

# Each field has candidate names; matching is case- and punctuation-insensitive.

# Raises an error with available columns if no candidate matches.


COLUMN_CANDIDATES = {
    # --- identifiers ---
    "contract_id": ["PIID", "Contract ID", "Award ID", "procurement_instrument_identifier"],

    # --- dates (used to build schedule targets) ---
    "award_date": ["Date Signed", "Award Date", "signeddate", "effective_date"],
    "start_date": ["Period Of Performance Start Date", "period_of_performance_start_date",
                   "Effective Date", "start_date"],
    "current_completion_date": ["Current Completion Date", "current_completion_date",
                                 "Period Of Performance Current End Date"],
    "ultimate_completion_date": ["Ultimate Completion Date", "ultimate_completion_date",
                                 "Period Of Performance Potential End Date"],

    # --- cost (used to build cost targets) ---
    "base_and_options_value": ["Base And All Options Value", "base_and_all_options_value",
                                "Base and Options Value"],
    "current_total_value": ["Current Total Value Of Award", "dollarsobligated",
                             "Action Obligation", "current_total_value_of_award"],

    # --- categorical / context features ---
    "contracting_agency": ["Contracting Agency ID", "contractingofficeagencyid", "Funding Agency"],
    "military_branch": ["Military Department", "contracting_department_name", "Component"],
    "naics_code": ["NAICS Code", "naics", "principal_naics_code"],
    "psc_code": ["Product Or Service Code", "PSC Code", "productorservicecode"],
    "contract_type": ["Type Of Contract", "contract_pricing", "type_of_contract"],
    "extent_competed": ["Extent Competed", "extentcompeted"],
    "number_of_offers": ["Number Of Offers Received", "number_of_offers_received"],
    "place_state": ["Place Of Performance State Code", "pop_state_code", "State"],
    "fiscal_year": ["Fiscal Year", "fiscal_year", "contractfiscal_year"],
}


COST_OVERRUN_THRESHOLD = 0.10     # >10% growth from base value = cost risk flag
SCHEDULE_OVERRUN_THRESHOLD_DAYS = 30  # >30 days beyond current vs ultimate completion date

MAX_MATURE_FISCAL_YEAR = 2017   # drop any project awarded after this year
TRAIN_YEARS_MAX = 2014          # train: <=2014, test: 2015-2017 (both mature)
