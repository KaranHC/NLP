import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, Any
import json
from dotenv import load_dotenv
# Import your CrewAI components
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI

from langchain_groq import ChatGroq
import os

from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool
)
# Configure Streamlit page
st.set_page_config(
    page_title="🚀 AI Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv("/home/aifruaduser/Hayagriva/whatsapp_agent/ava-whatsapp-agent-course/.env")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .analysis-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-success {
        background-color: #d1fae5;
        color: #065f46;
    }
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitStockAnalyzer:
    def __init__(self):
        self.setup_session_state()
        
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.SERPER_API_KEY = os.getenv('SERPER_API_KEY')  # For SerperDevTool
        self.llm_groq, self.llm_openai = self.setup_llm()
        # Initialize web scraping and search tools
        self.scrape_tool = ScrapeWebsiteTool()
        self.serper_tool = SerperDevTool(api_key=self.SERPER_API_KEY)

    def setup_session_state(self):
        """Initialize session state variables"""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []

    def setup_llm(self):
        """Setup LLM with API key from secrets"""
        try:
            # Initialize LLMs
            # API Keys
            
            llm_groq = ChatGroq(
                model="llama3-70b-8192",
                temperature=0.1,
                groq_api_key=self.GROQ_API_KEY
            )

            llm_openai = ChatOpenAI(
                model="gpt-4o",  # or "gpt-4-turbo", "gpt-3.5-turbo"
                temperature=0.2,
                openai_api_key=self.OPENAI_API_KEY,
                max_tokens=2000
            )
            return llm_groq, llm_openai
        except Exception as e:
            st.error(f"LLM setup failed: {e}")
            return None

    @tool("NSE Stock Data")
    def get_nse_stock_data(symbol: str) -> dict:
        """
        Get comprehensive NSE stock data for Indian equities.
        Input: NSE symbol (e.g., 'RELIANCE', 'TCS', 'INFY')
        """
        try:
            # Add .NS suffix for NSE stocks
            ticker = f"{symbol}.NS"
            stock = yf.Ticker(ticker)

            # Get comprehensive data
            info = stock.info
            history = stock.history(period="1y")
            financials = stock.financials
            balance_sheet = stock.balance_sheet

            return {
                "symbol": symbol,
                "current_price": info.get('currentPrice', 0),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "pb_ratio": info.get('priceToBook', 0),
                "dividend_yield": info.get('dividendYield', 0),
                "52_week_high": info.get('fiftyTwoWeekHigh', 0),
                "52_week_low": info.get('fiftyTwoWeekLow', 0),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "price_history": history.tail(30).to_dict(),
                "volume": info.get('volume', 0),
                "avg_volume": info.get('averageVolume', 0)
            }
        except Exception as e:
            return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

    @tool("BSE Stock Data")
    def get_bse_stock_data(symbol: str) -> dict:
        """
        Get BSE stock data for Indian equities.
        Input: BSE symbol (e.g., 'RELIANCE', 'TCS')
        """
        try:
            # Add .BO suffix for BSE stocks
            ticker = f"{symbol}.BO"
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="6mo")

            return {
                "symbol": symbol,
                "exchange": "BSE",
                "current_price": info.get('currentPrice', 0),
                "market_cap": info.get('marketCap', 0),
                "price_history": history.tail(20).to_dict(),
                "volume": info.get('volume', 0)
            }
        except Exception as e:
            return {"error": f"Failed to fetch BSE data for {symbol}: {str(e)}"}

    @tool("SEBI Compliance Check")
    def sebi_compliance_check(company_name: str) -> dict:
        """
        Check SEBI compliance and regulatory filings for Indian companies.
        Input: Company name
        """
        try:
            # Simulate SEBI compliance check
            compliance_data = {
                "company": company_name,
                "sebi_registered": True,
                "last_filing_date": "2024-03-31",
                "compliance_status": "Compliant",
                "pending_cases": 0,
                "insider_trading_alerts": [],
                "corporate_governance_score": 85,
                "disclosure_quality": "High"
            }
            return compliance_data
        except Exception as e:
            return {"error": f"Compliance check failed: {str(e)}"}

    @tool("FII DII Data")
    def get_fii_dii_data(symbol: str) -> dict:
        """
        Get Foreign Institutional Investor (FII) and Domestic Institutional Investor (DII) data.
        Input: Stock symbol
        """
        try:
            # Simulate FII/DII data
            fii_dii_data = {
                "symbol": symbol,
                "fii_holding_percent": 23.5,
                "dii_holding_percent": 15.2,
                "retail_holding_percent": 45.8,
                "promoter_holding_percent": 15.5,
                "fii_net_investment_30d": 150.5,  # in crores
                "dii_net_investment_30d": 89.3,
                "trend": "Positive FII inflow"
            }
            return fii_dii_data
        except Exception as e:
            return {"error": f"FII/DII data fetch failed: {str(e)}"}

    @tool("Indian Economic Indicators")
    def get_indian_economic_indicators() -> dict:
        """
        Get key Indian economic indicators affecting stock market.
        """
        try:
            indicators = {
                "repo_rate": 6.5,
                "inflation_rate": 5.2,
                "gdp_growth": 7.1,
                "fiscal_deficit": 5.8,
                "current_account_deficit": -2.1,
                "forex_reserves": 645.5,  # in billion USD
                "rupee_usd_rate": 83.25,
                "crude_oil_price": 85.5,
                "gold_price_inr": 62500,
                "nifty_pe": 22.5,
                "nifty_pb": 3.2,
                "market_sentiment": "Cautiously Optimistic"
            }
            return indicators
        except Exception as e:
            return {"error": f"Economic indicators fetch failed: {str(e)}"}

    @tool("Sector Analysis India")
    def analyze_indian_sector(sector: str) -> dict:
        """
        Analyze specific sector performance in Indian market.
        Input: Sector name (e.g., 'IT', 'Banking', 'Pharma', 'Auto')
        """
        try:
            sector_data = {
                "sector": sector,
                "performance_1m": 5.2,
                "performance_3m": 12.8,
                "performance_1y": 18.5,
                "top_performers": ["Stock1", "Stock2", "Stock3"],
                "laggards": ["Stock4", "Stock5"],
                "sector_pe": 25.3,
                "sector_pb": 3.8,
                "outlook": "Positive",
                "key_drivers": ["Government policies", "Global demand", "Technology adoption"],
                "risks": ["Regulatory changes", "Global slowdown", "Currency fluctuation"]
            }
            return sector_data
        except Exception as e:
            return {"error": f"Sector analysis failed: {str(e)}"}
    
    
    def create_agents_and_tasks(self, stock_symbol: str, investment_amount: float):
        """Create simplified agents and tasks for faster analysis"""

        # Indian Market Research Agent
        # Ensure your agent is properly configured
        indian_market_researcher = Agent(
            role="Indian Market Research Specialist",
            goal="""Conduct comprehensive research on Indian stock market trends,
            regulatory environment, FII/DII flows, and sector-specific analysis""",
            backstory="""You are a seasoned Indian equity research analyst with 15+ years
            of experience in NSE and BSE markets.""",
            tools=[
                self.get_nse_stock_data,
                self.get_bse_stock_data,
                self.get_fii_dii_data,
                self.get_indian_economic_indicators,
                self.analyze_indian_sector,
                self.scrape_tool,
                self.serper_tool
            ],
            llm=self.llm_openai,
            verbose=False,
            allow_delegation=False,
            max_iter=2,  # Add iteration limit
            max_execution_time=120  # Add time limit (5 minutes)
        )
        # Technical Analysis Agent
        technical_analyst = Agent(
            role="Technical Analysis Expert",
            goal="""Perform advanced technical analysis on Indian stocks using
            price action, volume analysis, and Indian market-specific patterns.
            Identify key support/resistance levels and provide entry/exit points.""",
            backstory="""You are a certified technical analyst (CFTe) with expertise
            in Indian market patterns. You understand the unique trading behaviors
            during Indian market hours, impact of global markets on Indian stocks,
            and seasonal patterns in Indian equities.""",
            tools=[self.get_nse_stock_data,self.get_bse_stock_data],
            llm=self.llm_openai,  # Using OpenAI Turbo for faster response
            verbose=False,
            max_iter=2,  # Add iteration limit
            max_execution_time=120,  # Add time limit (5 minutes)
            allow_delegation=False
        )

        # Financial Analysis Agent
        financial_analyst = Agent(
            role="Indian Corporate Finance Analyst",
            goal="""Analyze financial statements of Indian companies using Indian
            accounting standards (Ind AS), assess financial health, and evaluate
            valuation metrics specific to Indian market conditions.""",
            backstory="""You are a CFA charterholder with deep expertise in Indian
            corporate finance. You understand Indian accounting standards, tax
            implications, and sector-specific financial metrics. You can assess
            the impact of Indian regulatory changes on company financials.""",
            tools=[self.get_nse_stock_data, self.sebi_compliance_check],
            llm=self.llm_openai,  # Using OpenAI
            verbose=False,
            max_iter=2,  # Add iteration limit
            max_execution_time=120,  # Add time limit (5 minutes)
            allow_delegation=False
        )



        # Risk Assessment Agent
        risk_analyst = Agent(
            role="Indian Market Risk Analyst",
            goal="""Assess various risks including market risk, currency risk,
            regulatory risk, and liquidity risk specific to Indian equity investments.
            Provide risk-adjusted return analysis.""",
            backstory="""You are a risk management expert with FRM certification and
            deep understanding of Indian market risks. You can assess the impact of
            global events on Indian markets and provide comprehensive risk analysis.""",
            tools=[
                self.get_nse_stock_data,
                self.get_indian_economic_indicators,
                self.get_fii_dii_data,
                self.analyze_indian_sector
            ],
            llm=self.llm_openai,  # Using OpenAI
            verbose=False,
            max_iter=2,  # Add iteration limit
            max_execution_time=120,  # Add time limit (5 minutes)
            allow_delegation=False
        )

        # Portfolio Manager Agent
        portfolio_manager = Agent(
            role="Indian Equity Portfolio Manager",
            goal="""Synthesize all research, technical, financial, compliance, and risk
            analysis to provide actionable investment recommendations aligned with
            JP Morgan's investment philosophy and Indian market opportunities.""",
            backstory="""You are a senior portfolio manager at JP Morgan with 20+ years
            of experience managing Indian equity portfolios. You understand JP Morgan's
            investment philosophy and can make strategic decisions considering all
            aspects of Indian market dynamics.""",
            tools=[],
            llm=self.llm_openai,  # Using OpenAI for final recommendations
            verbose=False,
            max_iter=2,  # Add iteration limit
            max_execution_time=120,  # Add time limit (5 minutes)
            allow_delegation=False)

        # Quick Analysis Task
        market_research_task = Task(
        description=f"""
        Conduct comprehensive research on {stock_symbol} including:
        1. Current market position and competitive landscape
        2. Recent news and developments affecting the stock
        3. FII/DII investment patterns and institutional holdings
        4. Sector performance and outlook
        5. Government policies impact on the sector
        6. Economic indicators affecting the stock
        
        Focus on Indian market specifics including NSE/BSE performance,
        SEBI regulations, and domestic vs international factors.
        """,
        expected_output=f"""
        A detailed market research report for {stock_symbol} covering:
        - Current market position and key competitors
        - Recent news and developments (last 30 days)
        - FII/DII flow analysis
        - Sector outlook and government policy impact
        - Key risk factors and opportunities
        - Market sentiment analysis
        """,
        agent=indian_market_researcher
    )
    
        # Technical Analysis Task
        technical_analysis_task = Task(
            description=f"""
            Perform comprehensive technical analysis on {stock_symbol}:
            1. Price trend analysis (daily, weekly, monthly charts)
            2. Key support and resistance levels
            3. Volume analysis and trading patterns
            4. Technical indicators (RSI, MACD, Moving Averages)
            5. Chart patterns and breakout levels
            6. Entry and exit points for investment amount ₹{investment_amount:,.0f}
            
            Consider Indian market trading hours and volatility patterns.
            """,
            expected_output=f"""
            Technical analysis report for {stock_symbol} including:
            - Current trend direction and strength
            - Key support levels: [specific price levels]
            - Key resistance levels: [specific price levels]
            - Recommended entry price range
            - Stop-loss levels
            - Target price levels (3-6 months)
            - Risk-reward ratio analysis
            """,
            agent=technical_analyst,
            context=[market_research_task]
        )

        # Risk Assessment Task
        risk_assessment_task = Task(
            description=f"""
            Perform comprehensive risk analysis for investing ₹{investment_amount:,.0f} in {stock_symbol}:
            1. Market risk assessment (beta, volatility)
            2. Sector-specific risks
            3. Company-specific risks
            4. Liquidity risk analysis
            5. Currency and economic risks
            6. Regulatory and policy risks
            7. Risk-adjusted return calculation
            
            Provide risk mitigation strategies and position sizing recommendations.
            """,
            expected_output=f"""
            Risk assessment report for {stock_symbol} investment including:
            - Overall risk rating: [Low/Medium/High]
            - Market risk metrics: Beta, volatility, VaR
            - Key risk factors: [top 5 risks with impact assessment]
            - Liquidity analysis: [average daily volume vs investment size]
            - Recommended position size: [% of portfolio]
            - Risk mitigation strategies
            - Maximum acceptable loss: [amount and %]
            """,
            agent=risk_analyst,
            context=[market_research_task, technical_analysis_task]
        )

        # Financial Analysis Task
        financial_analysis_task = Task(
            description=f"""
            Analyze the financial health of {stock_symbol}:
            1. Revenue and profit growth trends
            2. Key financial ratios (P/E, P/B, ROE, ROA, Debt-to-Equity)
            3. Cash flow analysis
            4. Balance sheet strength
            5. Comparison with sector peers
            6. Valuation analysis using Indian market multiples

            Use Indian accounting standards (Ind AS) and consider
            sector-specific metrics relevant to Indian market.
            """,
            expected_output=f"""
            Financial analysis report for {stock_symbol} covering:
            - Revenue growth: [YoY and QoQ trends]
            - Profitability metrics: [margins and trends]
            - Key ratios: P/E, P/B, ROE, ROA, D/E
            - Cash flow analysis: [operating, investing, financing]
            - Balance sheet strength assessment
            - Peer comparison and sector ranking
            - Fair value estimation and current valuation assessment
            """,
            agent=financial_analyst,
            context=[market_research_task]
        )

        # Final Investment Recommendation Task
        investment_recommendation_task = Task(
            description=f"""
            Synthesize all analysis to provide final investment recommendation for {stock_symbol}:
            1. Consolidate research, technical, financial, compliance, and risk analysis
            2. Provide clear BUY/HOLD/SELL recommendation
            3. Suggest optimal investment strategy for ₹{investment_amount:,.0f}
            4. Define entry strategy, target prices, and exit conditions
            5. Align with JP Morgan's investment philosophy
            6. Consider Indian market dynamics and investor profile
            
            Provide actionable recommendations with clear rationale.
            """,
            expected_output=f"""
            FINAL INVESTMENT RECOMMENDATION for {stock_symbol}:
            
            RECOMMENDATION: [BUY/HOLD/SELL]
            CONFIDENCE LEVEL: [High/Medium/Low]
            
            INVESTMENT STRATEGY:
            - Recommended allocation: ₹[amount] ([% of total])
            - Entry price range: ₹[min] - ₹[max]
            - Target price (6 months): ₹[price] ([% upside])
            - Stop-loss level: ₹[price] ([% downside])
            - Investment horizon: [time period]
            
            KEY RATIONALE:
            - [3-5 key points supporting the recommendation]
            
            RISKS TO MONITOR:
            - [Top 3 risks that could affect the investment]
            
            EXECUTION PLAN:
            - [Step-by-step implementation strategy]
            
            JP MORGAN ALIGNMENT:
            - [How this aligns with JP Morgan's investment philosophy]
            """,
            agent=portfolio_manager,
            context=[market_research_task, technical_analysis_task, financial_analysis_task,risk_assessment_task ]
        )

        return [indian_market_researcher, technical_analyst, financial_analyst,risk_analyst, portfolio_manager], [market_research_task, technical_analysis_task,financial_analysis_task, risk_assessment_task, investment_recommendation_task]

    def run_analysis(self, stock_symbol: str, investment_amount: float):
        """Run the stock analysis"""
        try:
            agents, tasks = self.create_agents_and_tasks(stock_symbol, investment_amount)

            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
                memory=False
            )

            result = crew.kickoff(inputs={
                "stock_symbol": stock_symbol,
                "investment_amount": investment_amount
            })

            return {
                "status": "success",
                "result": result.raw if hasattr(result, 'raw') else str(result),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_quick_stock_data(self, symbol: str) -> dict:
        """Get essential stock data for analysis"""
        try:
            # Add .NS for Indian stocks, .BO for BSE
            if not symbol.endswith(('.NS', '.BO')):
                ticker = f"{symbol}.NS"
            else:
                ticker = symbol

            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="1y")

            return {
                "symbol": symbol,
                "current_price": info.get('currentPrice', 0),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "52_week_high": info.get('fiftyTwoWeekHigh', 0),
                "52_week_low": info.get('fiftyTwoWeekLow', 0),
                "sector": info.get('sector', 'Unknown'),
                "volume": info.get('volume', 0),
                "price_change": history['Close'].pct_change().iloc[-1] * 100 if len(history) > 0 else 0
            }
        except Exception as e:
            return {"error": f"Failed to fetch data: {str(e)}"}

    def display_stock_metrics(self, symbol: str):
        """Display real-time stock metrics"""
        try:
            stock_data = self.get_quick_stock_data(symbol)

            if "error" not in stock_data:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Current Price",
                        f"₹{stock_data['current_price']:.2f}",
                        f"{stock_data['price_change']:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Market Cap",
                        f"₹{stock_data['market_cap']/10000000:.0f}Cr" if stock_data['market_cap'] else "N/A"
                    )

                with col3:
                    st.metric("P/E Ratio", f"{stock_data['pe_ratio']:.2f}" if stock_data['pe_ratio'] else "N/A")

                with col4:
                    st.metric("Sector", stock_data['sector'])

                return stock_data
            else:
                st.error(stock_data['error'])
                return None

        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return None

    def display_analysis_results(self, results: Dict[str, Any]):
        """Display analysis results in an attractive format"""

        if results['status'] == 'success':
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 AI Analysis Results")

            # Parse and display the results
            analysis_text = results['result']
            st.markdown(f"```\n{analysis_text}\n```")

            # Add timestamp
            timestamp = datetime.fromisoformat(results['timestamp'])
            st.caption(f"Analysis completed at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")

    def create_price_chart(self, symbol: str):
        """Create an interactive price chart"""
        try:
            ticker = f"{symbol}.NS" if not symbol.endswith(('.NS', '.BO')) else symbol
            stock = yf.Ticker(ticker)
            data = stock.history(period="6mo")

            if not data.empty:
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol
                ))

                fig.update_layout(
                    title=f"{symbol} - 6 Month Price Chart",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    height=400,
                    showlegend=False
                )

                return fig
            else:
                return None

        except Exception as e:
            st.error(f"Error creating chart: {e}")
            return None

    def main(self):
        """Main application interface"""

        # Header
        st.title("🚀 AI Stock Analyzer Pro")
        st.markdown("*Powered by Advanced AI Agents for Comprehensive Stock Analysis*")

        # Sidebar inputs
        with st.sidebar:
            st.header("📊 Analysis Parameters")

            # Stock selection
            stock_symbol = st.text_input(
                "Stock Symbol",
                value="TCS",
                help="Enter NSE stock symbol (e.g., TCS, RELIANCE, INFY)"
            ).upper()

            # Investment amount
            investment_amount = st.number_input(
                "Investment Amount (₹)",
                min_value=1000,
                max_value=10000000,
                value=500000,
                step=10000,
                help="Enter your planned investment amount"
            )

            # Analysis button
            analyze_button = st.button(
                "🔍 Start AI Analysis",
                type="primary",
                use_container_width=True
            )

            st.divider()

            # Popular stocks
            st.subheader("💡 Popular Stocks")
            popular_stocks = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "LT", "BHARTIARTL"]

            for stock in popular_stocks:
                if st.button(stock, key=f"pop_{stock}", use_container_width=True):
                    st.session_state.selected_stock = stock
                    st.rerun()

        # Main content area
        if 'selected_stock' in st.session_state:
            stock_symbol = st.session_state.selected_stock

        # Display current stock metrics
        if stock_symbol:
            st.subheader(f"📈 {stock_symbol} - Live Metrics")
            stock_data = self.display_stock_metrics(stock_symbol)

            # Create two columns for chart and analysis
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 Price Chart")
                chart = self.create_price_chart(stock_symbol)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

            with col2:
                st.subheader("🎯 Quick Stats")
                if stock_data:
                    st.json({
                        "Symbol": stock_symbol,
                        "Sector": stock_data.get('sector', 'N/A'),
                        "52W High": f"₹{stock_data.get('52_week_high', 0):.2f}",
                        "52W Low": f"₹{stock_data.get('52_week_low', 0):.2f}",
                        "Volume": f"{stock_data.get('volume', 0):,}"
                    })

        # Analysis section
        if analyze_button and stock_symbol and self.llm_openai:
            st.session_state.analysis_running = True

            with st.spinner(f"🤖 AI agents analyzing {stock_symbol}... This may take 1-2 minutes"):
                progress_bar = st.progress(0)

                # Simulate progress
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)

                # Run analysis
                results = self.run_analysis(stock_symbol, investment_amount)
                st.session_state.analysis_results = results
                st.session_state.analysis_running = False

                # Store in history
                st.session_state.analysis_history.append({
                    "symbol": stock_symbol,
                    "amount": investment_amount,
                    "results": results
                })

                st.rerun()

        # Display results
        if st.session_state.analysis_results and not st.session_state.analysis_running:
            st.divider()
            self.display_analysis_results(st.session_state.analysis_results)

            # Download results
            if st.button("📥 Download Analysis Report"):
                report_data = {
                    "symbol": stock_symbol,
                    "investment_amount": investment_amount,
                    "analysis": st.session_state.analysis_results,
                    "generated_at": datetime.now().isoformat()
                }

                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"{stock_symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

        # Analysis history
        if st.session_state.analysis_history:
            st.divider()
            st.subheader("📋 Analysis History")

            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"{analysis['symbol']} - ₹{analysis['amount']:,} ({analysis['results']['timestamp'][:10]})"):
                    if analysis['results']['status'] == 'success':
                        st.text(analysis['results']['result'])
                    else:
                        st.error(analysis['results'].get('error', 'Analysis failed'))

# Run the app
if __name__ == "__main__":
    # Check for required secrets
    # if 'OPENAI_API_KEY' not in st.secrets:
    #     st.error("⚠️ OpenAI API key not found in secrets. Please add OPENAI_API_KEY to your Streamlit secrets.")
    #     st.stop()

    app = StreamlitStockAnalyzer()
    app.main()