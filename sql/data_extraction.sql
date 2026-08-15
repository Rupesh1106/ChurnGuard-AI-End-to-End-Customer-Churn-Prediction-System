-- ============================================
-- ChurnGuard AI — Data Extraction Queries
-- ============================================
-- SQL scripts for extracting and transforming
-- customer data for churn prediction.
-- ============================================

-- -------------------------------------------
-- 1. Extract Customer Demographics
-- -------------------------------------------
SELECT
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure,
    c.created_date,
    DATEDIFF(MONTH, c.created_date, GETDATE()) AS tenure_months
FROM
    customers c
WHERE
    c.is_active = 1 OR c.churn_date IS NOT NULL;


-- -------------------------------------------
-- 2. Extract Service Subscriptions
-- -------------------------------------------
SELECT
    c.customer_id,
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies
FROM
    customers c
    INNER JOIN services s ON c.customer_id = s.customer_id;


-- -------------------------------------------
-- 3. Extract Billing & Payment Information
-- -------------------------------------------
SELECT
    c.customer_id,
    b.contract_type,
    b.paperless_billing,
    b.payment_method,
    b.monthly_charges,
    b.total_charges,
    CASE
        WHEN b.total_charges > 0 AND c.tenure > 0
            THEN b.total_charges / c.tenure
        ELSE b.monthly_charges
    END AS avg_monthly_charge
FROM
    customers c
    INNER JOIN billing b ON c.customer_id = b.customer_id;


-- -------------------------------------------
-- 4. Full Customer Dataset for ML Pipeline
-- -------------------------------------------
SELECT
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure,
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    b.contract_type,
    b.paperless_billing,
    b.payment_method,
    b.monthly_charges,
    b.total_charges,
    CASE WHEN c.churn_date IS NOT NULL THEN 1 ELSE 0 END AS churn
FROM
    customers c
    INNER JOIN services s ON c.customer_id = s.customer_id
    INNER JOIN billing b ON c.customer_id = b.customer_id;


-- -------------------------------------------
-- 5. Churn Summary Statistics
-- -------------------------------------------
SELECT
    CASE WHEN churn_date IS NOT NULL THEN 'Churned' ELSE 'Active' END AS status,
    COUNT(*) AS customer_count,
    AVG(tenure) AS avg_tenure,
    AVG(b.monthly_charges) AS avg_monthly_charges,
    SUM(b.total_charges) AS total_revenue
FROM
    customers c
    INNER JOIN billing b ON c.customer_id = b.customer_id
GROUP BY
    CASE WHEN churn_date IS NOT NULL THEN 'Churned' ELSE 'Active' END;
