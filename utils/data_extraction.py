
def data_extract(path_to_db):
    # connect to the database
    conn = sqlite3.connect(path_to_db)
    
    query = """
        SELECT
    	e."Customer Identity",
    	"First Policy´s Year",
    	"Brithday Year",
    	"Educational Degree",
    	"Gross Monthly Salary",
    	"Geographic Living Area",
    	"Has Children (Y=1)",
    	"Customer Monetary Value",
    	"Claims Rate",
        l."Premiums in LOB: Motor",
    	l."Premiums in LOB: Household",
    	l."Premiums in LOB: Health",
    	l."Premiums in LOB:  Life",
    	l."Premiums in LOB: Work Compensations"
    	FROM
        Engage AS e
    	JOIN LOB AS l ON l."Customer Identity" = e."Customer Identity"
        ORDER BY
    	e."Customer Identity";
    """
    
    data_df = pd.read_sql_query(query, conn)
    df = data_df.copy()  # let's keep a copy of the original data
    
    #remaining column names to manageable variable names
    column_names = ['ID', 'First_Policy', 'Birthday', 'Education', 'Salary', 'Area', 'Children', 'CMV',
                    'Claims', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensation']
    #renaming the columns
    df.columns = column_names
    #seting 'ID' as index
    df.set_index('ID', inplace = True, drop = True)
    return data_df, df
