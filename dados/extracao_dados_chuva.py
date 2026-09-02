import pandas as pd
import basedosdados as bd

id_projeto = "pesquisa-ia-municipios"

query_consecutivos = """
WITH chuva_diaria AS (
    -- 1. Achata as horas em dias
    SELECT e.id_municipio, m.data, SUM(m.precipitacao_total) AS precip_dia
    FROM `basedosdados.br_inmet_bdmep.microdados` AS m
    JOIN `basedosdados.br_inmet_bdmep.estacao` AS e ON m.id_estacao = e.id_estacao
    WHERE m.ano = 2022 AND e.id_municipio LIKE '35%'
    GROUP BY e.id_municipio, m.data
),
marcacao_seca AS (
    -- 2. Transforma chuva em binário (1 para seco, 0 para chuva)
    SELECT id_municipio, data, precip_dia, IF(precip_dia = 0, 1, 0) AS seco
    FROM chuva_diaria
),
agrupamento_ilhas AS (
    -- 3. Cria "ilhas" cronológicas usando matemática de numeração de linhas
    SELECT id_municipio, seco,
           ROW_NUMBER() OVER(PARTITION BY id_municipio ORDER BY data) 
           - ROW_NUMBER() OVER(PARTITION BY id_municipio, seco ORDER BY data) AS grupo
    FROM marcacao_seca
),
contagem_blocos AS (
    -- 4. Conta o tamanho de cada bloco de dias secos consecutivos
    SELECT id_municipio, COUNT(*) AS dias_seguidos
    FROM agrupamento_ilhas
    WHERE seco = 1
    GROUP BY id_municipio, grupo
)
-- 5. Filtra para retornar apenas a maior sequência do ano por cidade
SELECT id_municipio, MAX(dias_seguidos) AS max_dias_consecutivos_seca
FROM contagem_blocos
GROUP BY id_municipio
"""

# Executando a consulta e salvando no DataFrame
df_teste = bd.read_sql(query_consecutivos, billing_project_id=id_projeto)

df_teste.to_csv('dias_consecutivos_seca_sp_2022.csv', index=False)

print("Sucesso! Olha os dados aqui:")