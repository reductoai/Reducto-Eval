from openai import OpenAI
import os
import pandas as pd
import csv
import reductoOCR

client = OpenAI(api_key='ENTER')

def query_gpt4(prompt, model="gpt-4-1106-preview"):

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers succinctly and directly."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def run_eval(qa_list, standard_df, reducto_df):
    questions, answers, gpt_answers, ocr_answers, ocr_context, reducto_answers, reducto_context = ([] for _ in range(7))

    for pair in qa_list:
        print("Starting next question")
        question = pair['Question']
        answer = pair['Answer']

        questions.append(question)
        answers.append(answer)

        #no context gpt prompt
        gpt_answer = query_gpt4(question)
        gpt_answers.append(gpt_answer)

        #ocr context gpt prompt
        context_indices = reductoOCR.search_most_similar_embedding(question, standard_df)
        print("Context Indices:", context_indices[:5]) 
        context_val = standard_df.loc[context_indices[0], 'Text_Chunk'] + "\n" + standard_df.loc[context_indices[1], 'Text_Chunk'] + "\n" + standard_df.loc[context_indices[2], 'Text_Chunk']
        #context_val = standard_df[standard_df['Chunk_Index'] == context_indices[0]]['Text_Chunk'].iloc[0] + "\n" + standard_df[standard_df['Chunk_Index'] == context_indices[1]]['Text_Chunk'].iloc[0] + "\n" + standard_df[standard_df['Chunk_Index'] == context_indices[2]]['Text_Chunk'].iloc[0]

        prompt = question + "\n Some information from a document that may be helpful for your response: \n" + context_val
        ocr_answer = query_gpt4(prompt)

        ocr_context.append(context_val)
        ocr_answers.append(ocr_answer)

        #reducto context gpt prompt
        context_indices = reductoOCR.search_most_similar_embedding(question, reducto_df)
        #context_val = reducto_df.loc[context_indices[0], 'Text_Chunk'] + "\n" + reducto_df.loc[context_indices[1], 'Text_Chunk'] 
        context_val = reducto_df.loc[context_indices[0], 'Text_Chunk'] + "\n" + reducto_df.loc[context_indices[1], 'Text_Chunk'] + "\n" + reducto_df.loc[context_indices[2], 'Text_Chunk']
       
        prompt = question + "\n Some information from a document that may be for helpful your response: \n" + context_val
        reducto_answer = query_gpt4(prompt)

        reducto_context.append(context_val)
        reducto_answers.append(reducto_answer)

    eval_data = zip(questions, answers, gpt_answers, ocr_answers, ocr_context, reducto_answers, reducto_context)
    return eval_data

apple_qa_list = [
    {'Question': 'According to the SEC\'s Rule 12b-2, what type of filer is Apple, Inc., and what implications does this classification have?',
     'Answer': 'Apple, Inc. is classified as a "large accelerated filer" according to the SEC\'s Rule 12b-2. This classification implies that Apple meets specific size criteria set by the SEC, such as market value and public float, and is subject to more stringent reporting and auditing requirements compared to smaller companies.'},

    {'Question': "What was the aggregate market value of the voting and non-voting stock held by non-affiliates of the Registrant (Apple) as of March 25, 2022?", 
     'Answer': "The aggregate market value was approximately $2,830,067,000,000."},

    {'Question': 'Has Apple, Inc. complied with the filing requirements of Sections 13 or 15(d) of the Securities Exchange Act of 1934 in the preceding year?',
     'Answer': 'Yes, Apple, Inc. has complied with the filing requirements. The 10-K filing indicates that the company has filed all reports required by Sections 13 or 15(d) of the Securities Exchange Act of 1934 during the preceding 12 months and has been subject to these filing requirements for the past 90 days.'},

    {'Question': 'How many shares of the Registrants (Apple) common stock were issued and outstanding as of October 14, 2022?',
     'Answer': 'There were 15,908,118,000 shares of common stock issued and outstanding.'},

    {'Question': 'What are the different models included in the iPhone line as of 2022?',
     'Answer': 'The iPhone line includes iPhone 14 Pro, iPhone 14, iPhone 13, iPhone SE, iPhone 12, and iPhone 11.'},
    
    {'Question': "List the products that fall under the 'Wearables, Home and Accessories' category for Apple.",
     'Answer': 'Under this category, the products include AirPods (AirPods, AirPods Pro, AirPods Max), Apple TV (Apple TV 4K, Apple TV HD), Apple Watch (Apple Watch Ultra, Apple Watch Series 8, Apple Watch SE), Beats products, HomePod mini, and other accessories.'},
    
    {'Question': 'What lawsuit did Epic Games file against Apple, and in which court was it filed?',
     'Answer': "Epic Games filed a lawsuit against the Company in the U.S. District Court for the Northern District of California, alleging violations of federal and state antitrust laws and California’s unfair competition law based on the Company's operation of its App Store."},
    
    {'Question': 'What was the outcome of the lawsuit between Epic Games and Apple on September 10, 2021?',
     'Answer': "On September 10, 2021, the Northern California District Court ruled in favor of the Company on nine out of ten counts in Epic’s claim, and in favor of the Company for its claims for breach of contract. However, it found that certain provisions of the Company's App Store Review Guidelines violate California’s unfair competition law and issued an injunction."},
    
    {'Question': 'As of October 14, 2022, how many shareholders of record were there at the Company (Apple)?',
     'Answer': 'As of October 14, 2022, the Company had 23,838 shareholders of record.'},

    {'Question': 'How many shares were purchased by the Company (Apple) between June 26, 2022, and July 30, 2022, and at what average price?',
     'Answer': '41,690 shares were purchased at an average price of $145.91.'},
    
    {'Question': 'During the period from August 28, 2022, to September 24, 2022, how many shares were purchased by the Company (Apple) and at what average price?',
     'Answer': '63,813 shares were purchased at an average price of $155.59.'},
    
    {'Question': 'What was the total number of shares purchased by the Company (Apple) from June 26,2022 September 24, 2022, and what is the approximate dollar value of shares that may yet be purchased under the plans or programs?',
     'Answer': 'A total of 160,172,000 shares were purchased, and the approximate dollar value of shares that may yet be purchased under the plans or programs is $60,665 million.'},
    
    {'Question': 'As of September 24, 2022, how much had been utilized from the $405 billion authorized by Apple\'s Board of Directors for the share repurchase program?',
     'Answer': 'As of September 24, 2022, $344.3 billion had been utilized from the authorized $405 billion for the share repurchase program.'},

     {'Question': "What was the value of Apple, Inc.'s stock in September 2017 and September 2022?",
     'Answer': 'In September 2017, the value was $100, and in September 2022, it was $411.'},
    
    {'Question': 'Compare the growth of Apple, Inc stock from September 2017 to September 2020 against the S&P 500 Index in the same period',
     'Answer': 'Apple, Inc grew from $100 to $303 while the S&P 500 Index grew from $100 to $142.'},

     {'Question': 'What was Apple\'s percentage increase in total net sales during fiscal 2022 compared to 2021, and what was the primary driver of this increase?',
     'Answer': 'Total net sales increased by 8% or $28.5 billion during 2022 compared to 2021, primarily driven by higher net sales of iPhone, Services, and Mac.'},
    
    {'Question': 'List some significant product, service, and software offerings announced by Apple in the first quarter of fiscal 2022.',
     'Answer': 'In the first quarter of fiscal 2022, significant announcements included the updated MacBook Pro 14” and MacBook Pro 16” with the Apple M1 Pro or M1 Max chip, and the third generation of AirPods.'},
    
    {'Question': 'What new products and updates did Apple announce in the third quarter of 2022?',
     'Answer': 'In the third quarter of 2022, the Company announced the updated MacBook Air and MacBook Pro 13” with the Apple M2 chip, iOS 16, macOS Ventura, iPadOS 16, watchOS 9, and Apple Pay Later service.'},
    
    {'Question': 'What changes did Apple announce to its Program authorization and quarterly dividend in April 2022?',
     'Answer': 'In April 2022, the Company announced an increase to its Program authorization from $315 billion to $405 billion and raised its quarterly dividend from $0.22 to $0.23 per share beginning in May 2022.'},
    
    {'Question': 'How much did Apple repurchase in common stock and pay in dividends and dividend equivalents during 2022?',
     'Answer': 'During 2022, the Company repurchased $90.2 billion of its common stock and paid dividends and dividend equivalents of $14.8 billion.'},

    {'Question': 'What percent of the total number of shares purchased by Apple were purchased as part of publicly announced plans or programs?',
     'Answer': '100% of purchased shares were part of a publicly announced plan or program.'},

    {'Question': 'What drove the increase in Apple\'s Americas net sales during 2022 compared to 2021?',
     'Answer': 'Americas net sales increased during 2022 compared to 2021 primarily due to higher net sales of iPhone, Services, and Mac.'},
    
    {'Question': 'What factors contributed to the increase in Apple\'s Europe net sales during 2022?',
     'Answer': 'Europe net sales increased during 2022 compared to 2021 due primarily to higher net sales of iPhone and Services, despite the weakness in foreign currencies relative to the U.S. dollar.'},
    
    {'Question': "How did currency fluctuations impact Apple\'s Greater China's net sales in 2022?",
     'Answer': "The strength of the renminbi relative to the U.S. dollar had a favorable year-over-year impact on Greater China's net sales during 2022."},
    
    {'Question': "Why did Apple\'s Japan's net sales decrease in 2022 compared to 2021?",
     'Answer': "Japan's net sales decreased during 2022 compared to 2021 due to the weakness of the yen relative to the U.S. dollar."},

    {'Question': 'What was the percentage change in iPhone net sales from 2021 to 2022, and what primarily contributed to this change?',
     'Answer': 'iPhone net sales increased by 7% from 2021 to 2022, primarily due to higher net sales from the new iPhone models released since the beginning of the fourth quarter of 2021.'},
    
    {'Question': 'How did iPad net sales perform in 2022 compared to 2021, and what was the main reason for this performance?',
     'Answer': 'iPad net sales decreased during 2022 compared to 2021, primarily due to lower net sales of iPad Pro.'},
    
    {'Question': 'What was the change in Mac net sales in 2022 compared to 2021, and what drove this change?',
     'Answer': 'Mac net sales increased during 2022 compared to 2021, primarily due to higher net sales of laptops.'},
    
    {'Question': 'Describe the performance of Apple\'s Wearables, Home and Accessories category in 2022 compared to 2021.',
     'Answer': 'Wearables, Home and Accessories net sales increased during 2022 compared to 2021, primarily due to higher net sales of Apple Watch and AirPods.'},
    
    {'Question': 'What factors led to the increase in Apple\'s Services net sales in 2022?',
     'Answer': 'Services net sales increased during 2022 compared to 2021, primarily due to higher net sales from advertising, cloud services, and the App Store.'},

    {'Question': "What is Apple's total balance of cash, cash equivalents, and unrestricted marketable securities as of September 24, 2022, and how does this affect its liquidity?",
     'Answer': "As of September 24, 2022, the Company's total balance was $156.4 billion. This, along with cash generated from operations and access to debt markets, is believed to be sufficient to satisfy its cash requirements and capital return program over the next 12 months and beyond."},
    
    {'Question': "What is the total amount of Apple's outstanding fixed-rate notes as of September 24, 2022, and what amount is payable within 12 months?",
     'Answer': "As of September 24, 2022, the Company had outstanding fixed-rate notes totaling $111.8 billion, with $11.1 billion payable within 12 months."},
    
    {'Question': "How much Commercial Paper did Apple have outstanding as of September 24, 2022, and what is its payment term?",
     'Answer': "As of September 24, 2022, the Company had $10.0 billion of Commercial Paper outstanding, all of which was payable within 12 months."},
    
    {'Question': "What are the Company's (Apple) payment obligations under its lease arrangements as of September 24, 2022?",
     'Answer': "As of September 24, 2022, the Company had fixed lease payment obligations of $15.3 billion, with $2.0 billion payable within 12 months."},
    
    {'Question': "What is the amount and nature of Apple’s manufacturing purchase obligations as of September 24, 2022?",
     'Answer': "As of September 24, 2022, the Company had manufacturing purchase obligations of $71.1 billion, with $68.4 billion payable within 12 months. These obligations are primarily noncancelable."}
    ]   
hoffman_qa_list = [
    
    {'Question': 'What is Gedatolisib and what is its code name?',
     'Answer': 'Gedatolisib is a dual inhibitor of PI3-K and mTOR, and its code name is PF-05212384, formerly known as PKI-587.'},
    
    {'Question': 'What is the US IND Number for Gedatolisib?',
     'Answer': 'The US IND Number for Gedatolisib is 128,914.'},

    {'Question': "What patient population is targeted in the clinical study 'CL-Gedatolisib-001'?",
     'Answer': 'The study targets previously untreated patients with ER+/HER2- Breast Cancer.'},

     {'Question': "What are the primary objectives of the Phase I study titled 'CL-Gedatolisib-001'?",
     'Answer': 'The primary objectives are to assess the safety, tolerability, and potential efficacy of Gedatolisib when used in combination with palbociclib and Faslodex in the neoadjuvant setting for previously untreated patients with ER+/HER2- breast cancer, and to determine the Maximum Tolerated Dose (MTD) of Gedatolisib in this combination.'},
    
    {'Question': "What is the secondary objective of the 'CL-Gedatolisib-001' study?",
     'Answer': 'The secondary objective is to evaluate the Pathologic Complete Response (pCR) induced by the Gedatolisib/palbociclib/Faslodex combination in the neoadjuvant setting for previously untreated patients with ER+/HER2- breast cancer.'},
    
    {'Question': "What are the exploratory objectives of the study 'CL-Gedatolisib-001'?",
     'Answer': 'The exploratory objectives are to assess the baseline values and potential correlations between these baseline values and response to the investigational neoadjuvant therapy, using genomic tests Foundation CDxTM in tumor tissue and FoundationOne®Liquid in peripheral whole blood.'},

     {'Question': "Note any differences in route and administration between Zoladex and Eligard in the study",
     'Answer': 'Both drugs are administered subcutaneously, but Zoladex is administered into the anterior abdominal wall below the navel while Eligard is administered into the abdomen, upper buttocks, or another location with adequate area not having excessive pigment, nodules, lesions, or hair that has not been recently used.'},

     {'Question': "List all modified Gedatolisib soses for Gedatolisib related toxicities if the starting dose is 215mg.",
     'Answer': 'The first dose reduction is 180mg, the second dose reduction is 150mg, and the third dose reduction is 140mg.'},

     {'Question': "When are fasting serum glucose, fasting triglycerides, insulin, C-peptide and cholesterol measured during the study?.",
     'Answer': 'Fasting serum glucose, fasting triglycerides, insulin, C-peptide and cholesterol are measured during the screening, on the first day of the first week, and before the surgery.'},

     {'Question': 'What type of study is being conducted and who are the participants?',
     'Answer': 'A dose-escalation Phase Ib clinical trial is being conducted in 18 patients with newly diagnosed Stage II-III ER+/HER2- breast cancer, who have not received prior therapy for their breast cancer and are intended to undergo surgery after four cycles of therapy.'},
    
    {'Question': 'Describe the treatment cycle and study drug administration for each patient.',
     'Answer': 'Each patient will be treated for four treatment cycles, each lasting four weeks or 28 days. During each cycle, Gedatolisib, Palbociclib, Faslodex, and in pre-menopausal patients, Zoladex, Eligard, or Lupron Depot, are administered as outlined in the study protocol.'},
    
    {'Question': 'Explain the dose escalation and de-escalation scheme of Gedatolisib in this study.',
     'Answer': 'Dose escalation of Gedatolisib is administered by IV once weekly for 16 weeks in three cohorts, with dose levels of 180 mg, 215 mg, and 260 mg. If no DLTs are observed, dose escalation continues. If a DLT is observed, the cohort expands to six patients. Dose de-escalation occurs if DLTs are observed in two or more patients, reducing the dose to 150 mg or discontinuing the study if necessary.'},
    
    {'Question': 'What is the definition of a Dose-Limiting Toxicity (DLT) in this study?',
     'Answer': 'A DLT is defined as any clinically relevant, grade ≥3 non-hematologic, non-infectious toxicity per National Cancer Institute Common Toxicity Criteria, excluding certain specific toxicities. It includes significant hematologic toxicities like specific grades of thrombocytopenia, neutropenia, and anemia.'},

     {'Question': 'What are the dose modification guidelines for metabolic toxicities in the study?',
     'Answer': 'For metabolic toxicities, no dose modification is required for Grade 1. For Grade ≥2 hyperglycemia, hyperglycemia management is implemented without dose modification. If Grade 4 hyperglycemia occurs despite optimal treatment, protocol directed therapy should be discontinued.'},
    
    {'Question': 'How should pneumonitis be managed in the study?',
     'Answer': 'For Grade 1 pneumonitis, no dose modification is required but appropriate therapy should be initiated. For Grade 2, therapy with Gedatolisib and Palbociclib may be interrupted, with dose reduction upon resumption. If Grade ≥ 2 pneumonitis recurs, protocol directed therapy should be discontinued. For Grade 3 pneumonitis, protocol directed therapy should be discontinued immediately.'},
    
    {'Question': 'What is the protocol for patients who fail to recover from drug-related toxicity?',
     'Answer': 'Patients must discontinue protocol directed therapy after failing to recover to Grade ≤1 or baseline severity for drug-related toxicity, or Grade ≤2 for toxicities not considered a safety risk, after a maximum delay of 2 weeks in initiating the next cycle.'},
    
    {'Question': 'How are adverse events related to Faslodex managed?',
     'Answer': 'For adverse events possibly related to Faslodex, the dose modification procedures stated in the Package Inserts of the drug will be followed.'},

    {'Question': 'What are the key inclusion criteria regarding the cancer stage and type for the clinical study?',
     'Answer': 'Patients must have Stage II-III, non-inflammatory invasive breast cancer with primary cancer in place, confirmed by core needle or incisional biopsy. The disease must be ER+ (ER expression ≥1%), HER2- (IHC staining 0 to 1+ or FISH ratio <2.0), previously untreated for breast cancer, operable, and the patient intends to undergo surgery after neoadjuvant therapy.'},
    
    {'Question': 'What are the specific requirements for tumor characteristics in the study?',
     'Answer': 'The tumor must be palpable or clinically assessable in the breast, radiographically measurable (longest diameter ≥10 mm), not axillary disease only, and can be multi-centric or bilateral, provided one lesion meets the criteria. Patients with lobular and luminal histology are included, but those with lobular histology should not exceed a quarter of the total number.'},
    
    {'Question': 'What is the route and administration and dose per administration for Lupron Depot in the study?',
     'Answer': 'Lupron Depot is administered throguh IM (intramuscular), using an aseptic technique according to the instructions stated in the Package Insert. The dose per administration in the study was 7.5mg.'},
    
    {'Question': 'What cardiac conditions would exclude a patient from the study?',
     'Answer': 'Conditions like marked baseline prolongation of QT/QTc interval, history of additional risk factors for torsade de pointes, arterial thrombotic event or stroke in the past 12 months, and uncontrolled hypertension or peripheral vascular disease ≥grade 2 exclude a patient.'},

    {'Question': 'What is the route and administration and dose per administration for Zoladex in the study?',
     'Answer': 'Zoladex is administered through SC (subcutaneous), into the anterior abdominal wall below the navel line using an aseptic technique under the supervision of a physician. The dose per administration in the study was 3.6mg.'},
    
    {'Question': 'What are the exclusion criteria related to bleeding and CNS conditions?',
     'Answer': 'Patients with active central nervous system metastasis, uncontrolled bleeding, a bleeding diathesis, or a history of significant bleeding in the past 6 months are excluded.'},
    
    {'Question': 'Are patients with diabetes eligible for the study?',
     'Answer': 'Patients with diabetes, except those with non-insulin dependent diabetes mellitus controlled with hemoglobin A1c ≤8%, are excluded.'},
    
    {'Question': 'What is the modified first dose reduction for Gedatolisib Related Toxicities due to an adverse event if the original Gedatolisib dose was 215mg?',
     'Answer': 'The first dose reduction is 180mg if the original dose was 215mg.'},

    {'Question': 'Which drugs are tested for Pre- and Post- Menopausal Subjects?',
     'Answer': 'In the study, Pre- and Post- Menopausal Subjects are treated with Gedastolib, Palbociclib, and Faslodex.'},

    {'Question': 'What is the route and administration and dose per administration for Palbociclib in the study?',
     'Answer': 'Palbociclib is administered through PO (oral) with food, and the dose per administration in the study was 125mg.'},

    {'Question': 'When are physical exam and body weight assessed during the study?',
     'Answer': 'Physical exam and body weight are performed during screening and immediately before Gedatolisib administration.'},

    {'Question': 'When are medical history & current medication assessed during the study?',
     'Answer': 'Medical history and current medication are assessed during the screening.'}
]
patent_qa_list = [

    {'Question': "Who are the inventors of the patent titled 'METHOD OF PRODUCING TUMOR ANTIBODIES' and what is its patent number?",
     'Answer': "The inventors are Hilary Koprowski and Carlo M. Croce, both from Pennsylvania. The patent number is 4,172,124."},
    
    {'Question': "What is the primary objective of the invention in this patent?",
     'Answer': "The primary objective of the invention is to produce antibodies that demonstrate specificity for malignant tumors. This is achieved by creating somatic cell hybrids between hypoxanthine phosphoribosyltransferase deficient myeloma cells and spleen or lymph cells derived from an animal previously primed with tumor cells."},
    
    {'Question': "What are the classifications and search fields of this patent?",
     'Answer': "The patent is classified under International Classification A61K 39/00 and A61K 39/42, and U.S. Classifications 424/85, 424/86, 435/240, 435/172. The field of search includes 424/85, 86, and 195/1.8."},
    
    {'Question': "What are the components of the fused cell hybrids used in this invention?",
     'Answer': "The fused cell hybrids used in this invention are composed of myeloma cells (malignant cells from primary tumors of bone marrow) and antibody-producing cells, preferably from the spleen or lymph nodes of animals immunized with tumor cells."},
    
    {'Question': "What prior research is referenced in relation to the development of this invention?",
     'Answer': "The patent references previous research by Kohler et al, Milstein et al, and Walsh, which described fused cell hybrids of spleen cells and myeloma cells. However, it was not known prior to this invention whether such hybrids could produce antibodies specific for tumors."},
    
    {'Question': "How are the antibodies produced and what types of tumor cells can they target?",
     'Answer': "Antibodies are produced by culturing and cloning the fused cell hybrids, and selecting clones that produce antibodies specific for tumor cells. These antibodies can target a range of human cancer cells, including melanoma, fibrosarcoma, breast carcinoma, lung carcinoma, colorectal carcinoma, and uterus carcinoma."},
    
    {'Question': "What is the importance of producing antibodies that react with tumors?",
     'Answer': "The production of antibodies that react with tumors is significant for analytical tools, diagnosis, immunotherapy, and medical research."},
    
    {'Question': "How are the somatic cell hybrids used to produce these antibodies formed?",
     'Answer': "The somatic cell hybrids are produced by fusing myeloma cells with anti-tumor antibody producing cells, preferably spleen or lymph cells from animals primed with tumor cells."},
    
    {'Question': "What is the preferred source of myeloma and anti-tumor antibody producing cells in this invention?",
     'Answer': "It is preferred to use the same species of animal as a source for both myeloma and anti-tumor antibody producing cells. A preferred cell line is a fused cell hybrid between tumor antigen primed mouse spleen cells and mouse myeloma cells."},
    
    {'Question': "How do the antibodies produced by different clones vary in their specificity for tumors?",
     'Answer': "Antibodies produced by different clones may vary in selectivity for tumor cells. Some may react only with cells of a specific tumor, while others may react with multiple tumor types. This variation is an important tool for medical research and determining cross-reactive specificities among different cell types."},

    {'Question': "Can the procedure for preparing hybrid cell lines be applied to other types of cells besides those mentioned in the example, and what can be used for immunization?",
     'Answer': "Yes, the procedure can be applied to other types of cells. For immunization, either tumor cells or cell fragments can be employed. Cell suspensions or cell fragment suspensions for injection can be prepared using established techniques."},

    {'Question': 'Describe the process of producing hybrid cells in this method.',
     'Answer': 'Hybrid cells are produced by fusing spleen cell suspensions, depleted of erythrocytes, with myeloma cells in the presence of polyethyleneglycol (PEG) 1000. After fusion, cells are suspended in hypoxanthine/aminopterin/thymidine (HAT) selective medium and seeded in flasks or tissue culture plates.'},

     {'Question': 'What is the purpose of Example I in the context of the patent?',
     'Answer': 'Example I serves as an illustrative example to demonstrate the practical application of the invention. It is not intended to limit the scope of the invention but to provide a clear understanding of how the method can be implemented.'},
    
    {'Question': 'What were the sources of spleen cells used for fusion with HPRT deficient P3×63 Ag8 mouse myeloma cells in Example I?',
     'Answer': 'In Example I, spleen cells were derived from BALB/c mice hyperimmunized with C57SV cells, C57BL/6J mice immunized with C57SV cells, and BALB/c mice immunized with MKSBu100.'},
    
    {'Question': 'What process was used for the fusion of myeloma and spleen cells in this example?',
     'Answer': 'Polyethyleneglycol (PEG) induced fusion was used for fusing HPRT deficient P3×63 Ag8 mouse myeloma cells with the spleen cells.'},
    
    {'Question': 'What are the abbreviations used in this example and what do they represent?',
     'Answer': 'Abbreviations like LN-SV, HT1080-6TG, HEK, F5-1, B1, C57SV, C57MEF, MKSBu100, BALB MEF, BALB 3T3, and HKBK-DNA-4 represent different cell lines such as SV40 transformed human fibroblasts, human fibrosarcoma derived cells, human embryo kidney cells, Syrian hamster fibroblasts, mouse strain C57BL fibroblasts, BALB/c kidney cells, etc.'},

     {'Question': 'How long after fusion did hybrid cells appear, and what was the subsequent process?',
     'Answer': 'Two to three weeks after fusion, hybrid cells growing in HAT selective medium appeared. These were then subcultured weekly in HAT selective medium.'},
    
    {'Question': 'How many independent hybrid cell cultures were obtained from the fusions, and what were they tested for?',
     'Answer': 'One hundred and forty-six independent hybrid cell cultures from 20 different fusions were obtained. They were tested for the production of antibodies specific for SV40 tumor antigen.'},
    
    {'Question': 'How many of the hybrid cell cultures produced antibodies against SV40 tumor antigen, and what was notable about them?',
     'Answer': 'Only 13 of the 146 hybrid cell cultures produced antibodies against SV40 tumor antigen, with ten of these independently derived from the same fusion experiment (B16.1).'},
    
    {'Question': 'What were the results of the antibody tests on various cell types as shown in Table 1?',
     'Answer': 'The antibodies produced by the hybridomas and a control mouse antiserum reacted with SV40 transformed human, hamster, and mouse cells, but did not react with normal or malignant cells derived from these same species, as determined by indirect immunofluorescence.'},

    {'Question': 'What does the table data from Table 1 indicate about the specificity of the antibodies?',
     'Answer': 'The table data shows that the antibodies, including those from different sources like culture fluid and serum from animals bearing tumors induced by hybrid cells, reacted positively with SV40 transformed cells (LN-SV, F5-1, C57SV, MKSBu100) but not with non-transformed cells (HT1080-6TG, HEK, B1, C57MEF, BALB MEF, BALB 3T3), indicating specificity for SV40 tumor antigen.'},

    {'Question': 'What test cells were used to assess the presence of anti-SV40 and BK virus tumor antigen antibodies?',
     'Answer': 'SV40 transformed human cells (LN-SV) and BK virus DNA transformed hamster kidney cells (HKBK-DNA-4) were used as test cells to assess the presence of anti-SV40 and BK virus tumor antigen antibodies.'},
    
    {'Question': 'How many hybridoma antibodies crossreacted with BK virus tumor antigen, and what was noted about the intensity of the fluorescence?',
     'Answer': 'Only four of the hybridoma antibodies crossreacted with BK virus tumor antigen. It was noted that the intensity of the fluorescence was generally weaker compared to the control serum.'},
    
    {'Question': 'What does Table 2 indicate about the specificity of the hybridoma antibodies for SV40 and BK virus tumor antigens?',
     'Answer': 'Table 2 shows that some hybridoma antibodies reacted positively with both SV40 and BK virus tumor antigens (LN-SV and HKBK-DNA-4), while others reacted only with SV40 tumor antigen. This indicates different antigenic determinants trigger antibody production, with only some being common to both SV40 and BK virus tumor antigens.'},

    {'Question': 'What is the significance of the antibody production triggered by different antigenic determinants?',
     'Answer': 'The significance lies in the fact that antibody production being triggered by different antigenic determinants is useful for their immunological and biochemical characterization, as only some determinants are common to both SV40 tumor antigen and BK virus tumor antigen.'},

    {'Question': 'What was the procedure followed in Example III for testing the hybridomas?',
     'Answer': 'In Example III, hybridomas (at least about 105 cells) were injected into a syngeneic host, which is the same strain of mouse from which the spleen cells were obtained. The injected cells were allowed to grow in the host for about 4 to 8 weeks.'},
    
    {'Question': 'What was measured after the injected cells grew in the host, and how were the results obtained?',
     'Answer': 'After the injected cells grew in the host for 4 to 8 weeks, the anti-SV40 antigen antibody titer of the serum and ascites were measured. The titers obtained from the serum, ascites, and the culture fluid are shown in Table 3.'},
    
    {'Question': 'What does Table 3 demonstrate regarding the antibody titers?',
     'Answer': 'Table 3 demonstrates that very high antibody titers were obtained from the hybridomas. It shows the titer levels in culture fluid, titer fluid, and ascites for various hybridomas.'},
    
    {'Question': 'What is the significance of the titer results obtained from the hybridomas in this example?',
     'Answer': 'The significance of the titer results is that they indicate the successful production of high levels of anti-SV40 antigen antibodies in the hybridomas, validating the effectiveness of the method for producing specific antibodies.'},

    {'Question': 'What procedure was used to immunize mice in Example IV, and what cells were used for the immunization?',
     'Answer': 'Mice were immunized using cells from human melanomas and colorectal carcinomas, and hybrid cultures between human melanoma and mouse fibroblast cells. The immunization involved a primary intraperitoneal injection of live tumor cells and a secondary intravenous booster, followed by spleen cell extraction for hybrid culture formation.'},
    
    {'Question': 'What were the results of fusing spleen cells with human melanoma and colorectal carcinoma cells?',
     'Answer': 'Of 29 hybrid cultures obtained from fusing spleen cells with human melanoma cells, 9 secreted antibodies reacting with human melanoma in radioimmunoassay. With colorectal carcinoma cells, 3 out of 8 cultures produced anti-colorectal carcinoma antibodies.'},
    
    {'Question': 'What does Table 4 indicate about the crossreactivity of antibodies produced against melanoma and colorectal carcinoma?',
     'Answer': 'Table 4 indicates that some hybridomas produced antibodies that reacted with melanoma and colorectal carcinoma cells, as well as with normal human cells, demonstrating varied crossreactivity.'},
    
    {'Question': 'Were there any hybridomas that produced antibodies specific only to melanoma cells?',
     'Answer': 'Yes, one anti-melanoma antibody producing hybridoma (#13) reacted only against melanoma cells and not against colorectal carcinoma or normal human cells. Another hybridoma (#6) reacted against all melanomas tested but not against normal human cells.'},
    
    {'Question': 'What types of cells are preferred for producing tumor antibodies according to the patent claims?',
     'Answer': 'Spleen cells and lymph node cells are preferred for producing tumor antibodies.'},
    
    {'Question': 'Which animals are specified for immunization in the method of producing malignant tumor antibodies?',
     'Answer': 'Mice and rats are specified as the animals for immunization in the method.'},
    
    {'Question': 'What specific type of cancer cells are mentioned for immunization in the claims?',
     'Answer': 'Human cancer cells, specifically melanoma and colorectal carcinoma, are mentioned for immunization.'},
    
]


if(os.path.exists('reducto_apple_dataframe2.pkl')):
    reducto_df = pd.read_pickle('reducto_apple_dataframe2.pkl')
if(os.path.exists('apple_dataframe.pkl')):
    standard_df = pd.read_pickle('apple__dataframe.pkl')

zipped_outputs = run_eval(patent_qa_list, standard_df, reducto_df)
print("Ran Eval")
csv_file_path = 'patest_output.csv'

with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Original Question", "Original Answer", "No Context GPT Answers", "OCR Context GPT Answers", "OCR Context Val", "Reducto Context GPT Answers", "Reducto Context Val"])  # Column headers
    writer.writerows(zipped_outputs)




zipped_outputs = run_eval(patent_qa_list, standard_df, reducto_df)
print("Ran Eval")
csv_file_path = 'patest_output.csv'

with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Original Question", "Original Answer", "No Context GPT Answers", "OCR Context GPT Answers", "OCR Context Val", "Reducto Context GPT Answers", "Reducto Context Val"])  # Column headers
    writer.writerows(zipped_outputs)