from sqlalchemy import create_engine

# 数据库连接信息
DATABASE_URL = "mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
