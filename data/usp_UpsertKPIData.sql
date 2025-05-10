USE [ABC]
GO
/****** Object:  StoredProcedure [dbo].[usp_UpsertKPIData]    Script Date: 2025-05-08 11:13:28 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- Procedure to upsert KPI values
ALTER PROCEDURE [dbo].[usp_UpsertKPIData]
    @PunchCodeID INT,
    @KPIDate DATE,
    @KPIValue DECIMAL(18,2),
    @Username NVARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @KPIDataID INT;
    DECLARE @OldValue DECIMAL(18,2);
    DECLARE @ChangeType NVARCHAR(20);
    
    -- Check if the record exists
    SELECT @KPIDataID = KPIDataID, @OldValue = KPIValue
    FROM KPIData
    WHERE PunchCodeID = @PunchCodeID AND KPIDate = @KPIDate AND IsActive = 1;
    
    -- If record exists, update it
    IF @KPIDataID IS NOT NULL
    BEGIN
        UPDATE KPIData
        SET KPIValue = @KPIValue,
            ModifiedBy = @Username,
            ModifiedDate = GETDATE()
        WHERE KPIDataID = @KPIDataID;
        
        SET @ChangeType = 'UPDATE';
    END
    -- If record doesn't exist, insert it
    ELSE
    BEGIN
        INSERT INTO KPIData (PunchCodeID, KPIDate, KPIValue, CreatedBy)
        VALUES (@PunchCodeID, @KPIDate, @KPIValue, @Username);
        
        SET @KPIDataID = SCOPE_IDENTITY();
        SET @ChangeType = 'INSERT';
    END
    
    -- Log the change
    INSERT INTO KPIChangeLog (KPIDataID, PunchCodeID, KPIDate, OldValue, NewValue, ChangeType, ChangedBy)
    VALUES (@KPIDataID, @PunchCodeID, @KPIDate, @OldValue, @KPIValue, @ChangeType, @Username);
    
    RETURN @KPIDataID;
END
