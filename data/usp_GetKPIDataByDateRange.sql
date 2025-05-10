USE [ABC]
GO

/****** Object:  StoredProcedure [dbo].[usp_GetKPIDataByDateRange]    Script Date: 2025-05-08 11:12:16 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Create updated stored procedure
CREATE PROCEDURE [dbo].[usp_GetKPIDataByDateRange]
    @StartDate DATE,
    @EndDate DATE,
    @PeriodType NVARCHAR(10) = 'DAILY' -- DAILY, WEEKLY, MONTHLY
AS
BEGIN
    SET NOCOUNT ON;
    
    -- DAILY: Return daily data directly
    IF @PeriodType = 'DAILY'
    BEGIN
        -- Return daily KPI values with proper date and punch code
        SELECT 
            k.KPIDate,
            p.PunchCodeValue,
            k.KPIValue
        FROM 
            KPIData k
        JOIN 
            PunchCodes p ON k.PunchCodeID = p.PunchCodeID
        WHERE 
            k.KPIDate BETWEEN @StartDate AND @EndDate
            AND k.IsActive = 1
        ORDER BY 
            k.KPIDate, p.PunchCodeValue;
    END
    
    -- WEEKLY: Calculate and return weekly averages
    ELSE IF @PeriodType = 'WEEKLY'
    BEGIN
        -- Generate all weeks in the date range
        WITH DateSequence AS (
            -- Start from the beginning of the week containing the start date
            SELECT DATEADD(DAY, -(DATEPART(WEEKDAY, @StartDate)-1), @StartDate) AS WeekStart
            UNION ALL
            -- Generate subsequent weeks
            SELECT DATEADD(DAY, 7, WeekStart)
            FROM DateSequence
            WHERE DATEADD(DAY, 7, WeekStart) <= @EndDate
        ),
        WeekRanges AS (
            SELECT 
                WeekStart,
                DATEADD(DAY, 6, WeekStart) AS WeekEnd
            FROM DateSequence
        )
        
        -- Calculate weekly averages
        SELECT 
            CONVERT(VARCHAR(10), wr.WeekStart, 120) + ' to ' + CONVERT(VARCHAR(10), wr.WeekEnd, 120) AS WeekRange,
            p.PunchCodeValue,
            AVG(CAST(k.KPIValue AS FLOAT)) AS AverageKPIValue
        FROM 
            WeekRanges wr
        CROSS JOIN 
            PunchCodes p
        LEFT JOIN 
            KPIData k ON k.KPIDate BETWEEN wr.WeekStart AND wr.WeekEnd
                      AND k.PunchCodeID = p.PunchCodeID
                      AND k.IsActive = 1
        WHERE
            p.IsActive = 1
        GROUP BY
            wr.WeekStart, wr.WeekEnd, p.PunchCodeValue
        ORDER BY
            wr.WeekStart, p.PunchCodeValue;
    END
    
    -- MONTHLY: Calculate and return monthly averages
    ELSE IF @PeriodType = 'MONTHLY'
    BEGIN
        -- Generate all months in the date range
        WITH Months AS (
            -- Start from the beginning of the month containing the start date
            SELECT DATEFROMPARTS(YEAR(@StartDate), MONTH(@StartDate), 1) AS MonthStart
            UNION ALL
            -- Generate subsequent months
            SELECT DATEADD(MONTH, 1, MonthStart)
            FROM Months
            WHERE DATEADD(MONTH, 1, MonthStart) <= @EndDate
        )
        
        -- Calculate monthly averages
        SELECT 
            FORMAT(m.MonthStart, 'yyyy-MM') AS MonthYearStr,
            p.PunchCodeValue,
            AVG(CAST(k.KPIValue AS FLOAT)) AS AverageKPIValue
        FROM 
            Months m
        CROSS JOIN 
            PunchCodes p
        LEFT JOIN 
            KPIData k ON k.KPIDate BETWEEN m.MonthStart AND EOMONTH(m.MonthStart)
                      AND k.PunchCodeID = p.PunchCodeID
                      AND k.IsActive = 1
        WHERE
            p.IsActive = 1
        GROUP BY
            m.MonthStart, p.PunchCodeValue
        ORDER BY
            m.MonthStart, p.PunchCodeValue;
    END
    
    -- Invalid period type
    ELSE
    BEGIN
        RAISERROR('Invalid @PeriodType parameter. Use DAILY, WEEKLY, or MONTHLY.', 16, 1);
        RETURN;
    END
END
GO

