# ---------------------------------------------------------------------------
# ADDITIONAL symptom classification for the RSV/SPN analysis.
# Adds ari_case, any_symptom and symptom_group.
# Does NOT modify Symptom_status, which is kept as the original variable.
#
# Three changes from the original draft, each marked [FIX] below.
# ---------------------------------------------------------------------------

library(dplyr)

# [FIX 1] Treat a missing symptom field as "not reported" rather than NA.
#         In the original, `Cough with sputum == "Yes"` is NA for the 83 children
#         where that field is blank, and `FALSE | NA` is NA in R, so any_symptom
#         (and therefore symptom_group) came back NA for some children.
yes <- function(x) !is.na(x) & x == "Yes"

data <- data %>%
  mutate(
    symptom_start_date = as.Date(
      paste(`Date Day symptoms started`,
            `Month symptoms started`,
            `Year symptoms started`),
      format = "%d %b %Y"
    ),

    # [FIX 2] Use `sample date`, not `Date`. The two disagree for 3 children
    #         (rows where Date is 2024-11-01 / 2024-01-31 but sample date is
    #         2024-10-31 / 2024-01-21). `sample date` is the one used elsewhere
    #         in the analysis, so this keeps the onset window consistent.
    collection_date = as.Date(`sample date`),

    days_since_onset = as.numeric(collection_date - symptom_start_date),
    symptoms_within_month = !is.na(days_since_onset) &
      days_since_onset >= 0 & days_since_onset <= 30,

    step1 = yes(`Clinical suspicion of Acute Respiratory Infection`) &
      symptoms_within_month,

    major_symptom =
      yes(`A history of self-reported feverishness or measured fever of ≥ 38°C`) |
      yes(Cough) |
      yes(`Shortness of breath OR Tachypnoea`) |
      yes(`Loss of smell`) |
      yes(`Loss of taste`),

    additional_symptom =
      yes(`Sore Throat`) |
      yes(`Runny nose (rhinorrhea)`) |
      yes(Headache) |
      yes(`Muscle aches (myalgia)`) |
      yes(`Fatigue / Malaise`) |
      yes(`Chest pain`),

    ari_case = factor(
      ifelse(step1 & major_symptom & additional_symptom, "Yes", "No"),
      levels = c("No", "Yes")
    )
  )

data <- data %>%
  mutate(
    any_symptom = major_symptom | additional_symptom |
      yes(`Cough with sputum`) | yes(`Cough with hemoptysis`) |
      yes(Wheezing) | yes(`Joint pain (arthralgia)`) |
      yes(`Inability to walk`) | yes(`Lower chest wall indrawing`) |
      yes(`Altered consciousness / confusion`) | yes(Seizures) |
      yes(`Abdominal pain`) | yes(`Vomiting / Nausea`) |
      yes(Diarrhoea) | yes(Conjunctivitis) | yes(`Skin rash`) |
      yes(`Skin ulcers`) | yes(Lymphadenopathy) | yes(`Other symptom`),

    symptom_group = case_when(
      ari_case == "Yes"                 ~ "ARI",
      ari_case == "No" & any_symptom    ~ "Symptomatic Non-ARI",
      TRUE                              ~ "Asymptomatic"      # [FIX 3] catch-all,
      # so no child is left NA when any_symptom is FALSE
    ) %>% factor(levels = c("ARI", "Symptomatic Non-ARI", "Asymptomatic"))
  )

# --- what this yields on the 339-child dataset -----------------------------
# ari_case:       Yes 117, No 222
# symptom_group:  ARI 117, Symptomatic Non-ARI 218, Asymptomatic 4
#
# NOTE: only 4 of 339 children report no symptom of any kind, so the
# "Asymptomatic" level of symptom_group is too small to analyse. Use
# ari_case (117 vs 222) for any symptom contrast.
#
# Four columns are constant "No" throughout and contribute nothing:
# Sore Throat, Joint pain (arthralgia), Seizures, Lymphadenopathy.

table(data$ari_case)
table(data$symptom_group)
table(data$Symptom_status, data$ari_case)   # agreement with the original label
