"""
PDF Report Generator
Generates detailed PDF reports for trading performance.
"""
import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from io import BytesIO
import base64
from fpdf import FPDF
from pathlib import Path

from config.config import (
    REPORTS_DIR, SYMBOL, INITIAL_CAPITAL, 
    GENERATE_PDF_REPORTS, PDF_REPORT_FREQUENCY
)
from reporting.metrics import PerformanceMetrics
from utils.logger import setup_logger

logger = setup_logger("pdf_generator")

# Set up matplotlib
mpl.use('Agg')  # Use non-interactive backend
plt.style.use('dark_background')  # Use dark theme for plots

class PDF(FPDF):
    """
    Custom PDF class with header and footer.
    """
    def __init__(self, title="FX Trading Bot Report"):
        super().__init__()
        self.title = title
        self.set_author("FX Trading Bot")
        self.set_creator("FX Trading Bot")
        self.set_subject("Trading Report")
        self.logo_path = None  # Path to logo image
    
    def header(self):
        # Logo (if available)
        if self.logo_path and os.path.exists(self.logo_path):
            self.image(self.logo_path, 10, 8, 20)
            logo_offset = 35
        else:
            logo_offset = 10
        
        # Title
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, self.title, 0, 1, 'C')
        
        # Date
        self.set_font('helvetica', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        
        # Line break
        self.ln(5)
    
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Page number
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def chapter_title(self, title):
        # Add a chapter title
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(0, 51, 102)  # Navy blue
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    
    def section_title(self, title):
        # Add a section title
        self.set_font('helvetica', 'B', 12)
        self.set_text_color(0, 76, 153)  # Lighter blue
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
    
    def body_text(self, text):
        # Add body text
        self.set_font('helvetica', '', 10)
        self.set_text_color(0, 0, 0)  # Black
        self.multi_cell(0, 5, text)
        self.ln(5)
    
    def add_metric(self, name, value, width=60, color=(0, 0, 0)):
        # Add a metric with name and value
        self.set_font('helvetica', 'B', 10)
        self.set_text_color(*color)
        self.cell(width, 6, name, 0, 0, 'L')
        
        self.set_font('helvetica', '', 10)
        self.cell(0, 6, str(value), 0, 1, 'L')
    
    def add_image(self, img_path, w=0, h=0, caption=None):
        # Add an image with optional caption
        self.image(img_path, x=None, y=None, w=w, h=h)
        
        if caption:
            self.set_font('helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, caption, 0, 1, 'C')
            
        self.ln(5)
    
    def add_plot(self, figure, w=0, h=0, caption=None):
        # Convert matplotlib figure to image and add it
        img_stream = BytesIO()
        figure.savefig(img_stream, format='png', bbox_inches='tight', dpi=300)
        img_stream.seek(0)
        
        # Create a temporary file
        temp_file = f"temp_plot_{int(time.time())}.png"
        with open(temp_file, 'wb') as f:
            f.write(img_stream.read())
        
        # Add image
        self.add_image(temp_file, w, h, caption)
        
        # Clean up
        plt.close(figure)
        os.remove(temp_file)
    
    def add_table(self, headers, data, width=None):
        # Add a table with headers and data
        if width is None:
            width = 190 / len(headers)
        
        # Headers
        self.set_font('helvetica', 'B', 10)
        self.set_fill_color(240, 240, 240)  # Light gray
        
        for header in headers:
            self.cell(width, 7, header, 1, 0, 'C', 1)
        self.ln()
        
        # Data
        self.set_font('helvetica', '', 9)
        self.set_fill_color(255, 255, 255)  # White
        
        alt_row = False
        for row in data:
            if alt_row:
                self.set_fill_color(245, 245, 245)  # Very light gray
            else:
                self.set_fill_color(255, 255, 255)  # White
            
            for cell in row:
                self.cell(width, 6, str(cell), 1, 0, 'C', 1)
            self.ln()
            alt_row = not alt_row
        
        self.ln(5)

class ReportGenerator:
    """
    Generates detailed PDF reports for trading performance.
    """
    def __init__(self, 
                 metrics: Optional[PerformanceMetrics] = None,
                 trade_log: Optional[List[Dict[str, Any]]] = None,
                 equity_curve: Optional[List[float]] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the ReportGenerator.
        
        Args:
            metrics (Optional[PerformanceMetrics], optional): Performance metrics object. 
                Defaults to None (creates a new one).
            trade_log (Optional[List[Dict[str, Any]]], optional): Trade log. 
                Defaults to None.
            equity_curve (Optional[List[float]], optional): Equity curve. 
                Defaults to None.
            output_dir (Optional[str], optional): Output directory for reports. 
                Defaults to None (uses REPORTS_DIR from config).
        """
        self.metrics = metrics or PerformanceMetrics()
        self.trade_log = trade_log or []
        self.equity_curve = equity_curve or [INITIAL_CAPITAL]
        self.output_dir = output_dir or Path(REPORTS_DIR)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Dark mode colors for plots
        self.colors = {
            'background': '#121212',
            'text': '#FFFFFF',
            'grid': '#333333',
            'accent': '#BB86FC',
            'accent2': '#03DAC6',
            'profit': '#00FF00',
            'loss': '#FF0000'
        }
    
    def update_trade_log(self, trade_log: List[Dict[str, Any]]) -> None:
        """
        Update the trade log.
        
        Args:
            trade_log (List[Dict[str, Any]]): Trade log.
        """
        self.trade_log = trade_log
        self.metrics.update_trade_log(trade_log)
    
    def update_equity_curve(self, equity_curve: List[float]) -> None:
        """
        Update the equity curve.
        
        Args:
            equity_curve (List[float]): Equity curve.
        """
        self.equity_curve = equity_curve
        self.metrics.update_equity_curve(equity_curve)
    
    def generate_report(self, 
                       title: str = "FX Trading Bot Performance Report",
                       period: str = "all",
                       include_trades: bool = True,
                       report_type: str = "standard") -> str:
        """
        Generate a PDF report.
        
        Args:
            title (str, optional): Report title. Defaults to "FX Trading Bot Performance Report".
            period (str, optional): Report period. Defaults to "all". Options: "all", "daily", "weekly", "monthly".
            include_trades (bool, optional): Whether to include all trades. Defaults to True.
            report_type (str, optional): Report type. Defaults to "standard". Options: "standard", "detailed", "summary".
            
        Returns:
            str: Path to the generated PDF file.
        """
        # Calculate metrics if not already done
        if not self.metrics.metrics:
            self.metrics.calculate_metrics()
        
        # Create PDF
        pdf = PDF(title)
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Generate report sections based on type
        if report_type == "summary":
            self._add_summary_section(pdf)
        elif report_type == "detailed":
            self._add_summary_section(pdf)
            self._add_performance_section(pdf)
            self._add_strategy_performance_section(pdf)
            self._add_equity_curve_section(pdf)
            
            if include_trades:
                self._add_trades_section(pdf)
                
            self._add_distribution_section(pdf)
            self._add_monthly_performance_section(pdf)
        else:  # standard
            self._add_summary_section(pdf)
            self._add_performance_section(pdf)
            self._add_equity_curve_section(pdf)
            
            if include_trades:
                self._add_trades_section(pdf, limit=20)  # Limit to 20 trades
        
        # Generate file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"report_{timestamp}.pdf")
        
        # Save PDF
        pdf.output(report_file)
        
        logger.info(f"Generated PDF report: {report_file}")
        
        return report_file
    
    def _add_summary_section(self, pdf: PDF) -> None:
        """
        Add summary section to the PDF.
        
        Args:
            pdf (PDF): The PDF object.
        """
        pdf.chapter_title("Summary")
        
        # Summary text
        metrics = self.metrics.metrics
        
        total_return = metrics.get("total_return_pct", 0)
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 0)
        
        # Determine text color based on total return
        text_color = (0, 128, 0) if total_return >= 0 else (128, 0, 0)  # Green or red
        
        summary_text = f"This report summarizes the trading performance for {SYMBOL}. "
        summary_text += f"The bot has achieved a total return of {total_return:.2f}% with a {win_rate:.2f}% win rate. "
        summary_text += f"It has a profit factor of {profit_factor:.2f} and a Sharpe ratio of {sharpe_ratio:.2f}. "
        summary_text += f"The maximum drawdown was {max_drawdown:.2f}%."
        
        pdf.body_text(summary_text)
        
        # Key metrics in a grid
        metrics_to_show = [
            ("Initial Capital", f"${metrics.get('initial_capital', 0):,.2f}"),
            ("Current Capital", f"${metrics.get('current_capital', 0):,.2f}"),
            ("Total Return", f"{total_return:.2f}%"),
            ("Total Profit", f"${metrics.get('total_profit', 0):,.2f}"),
            
            ("Total Trades", metrics.get("total_trades", 0)),
            ("Winning Trades", metrics.get("winning_trades", 0)),
            ("Losing Trades", metrics.get("losing_trades", 0)),
            ("Win Rate", f"{win_rate:.2f}%"),
            
            ("Profit Factor", f"{profit_factor:.2f}"),
            ("Sharpe Ratio", f"{sharpe_ratio:.2f}"),
            ("Max Drawdown", f"{max_drawdown:.2f}%"),
            ("Total Pips", f"{metrics.get('total_pips', 0):,.1f}")
        ]
        
        pdf.ln(5)
        
        # Create two columns
        col_width = 95
        for i, (name, value) in enumerate(metrics_to_show):
            if i % 2 == 0:
                pdf.set_x(10)
            else:
                pdf.set_x(105)
            
            # Use color for total return
            if name == "Total Return":
                pdf.add_metric(name, value, col_width, text_color)
            else:
                pdf.add_metric(name, value, col_width)
            
            if i % 2 == 1:
                pdf.ln(6)
        
        pdf.ln(10)
    
    def _add_performance_section(self, pdf: PDF) -> None:
        """
        Add performance section to the PDF.
        
        Args:
            pdf (PDF): The PDF object.
        """
        pdf.chapter_title("Performance Metrics")
        
        # Create performance charts
        fig = self._create_performance_charts()
        pdf.add_plot(fig, caption="Performance Metrics")
    
    def _add_strategy_performance_section(self, pdf: PDF) -> None:
        """
        Add strategy performance section to the PDF.
        
        Args:
            pdf (PDF): The PDF object.
        """
        metrics = self.metrics.metrics
        strategy_performance = metrics.get("strategy_performance", {})
        
        if not strategy_performance:
            return
        
        pdf.chapter_title("Strategy Performance")
        
        # Create table data
        headers = ["Strategy", "Trades", "Win Rate", "P&L", "Pips"]
        data = []
        
        for strategy, stats in strategy_performance.items():
            win_rate = stats.get("win_rate", 0)
            trades = stats.get("trades", 0)
            pnl = stats.get("pnl", 0)
            pips = stats.get("pips", 0)
            
            data.append([
                strategy,
                trades,
                f"{win_rate:.2f}%",
                f"${pnl:.2f}",
                f"{pips:.1f}"
            ])
        
        # Add table
        pdf.add_table(headers, data)
        
        # Create strategy comparison chart
        fig = self._create_strategy_comparison_chart()
        pdf.add_plot(fig, caption="Strategy Comparison")
    
    def _add_equity_curve_section(self, pdf: PDF) -> None:
        """
        Add equity curve section to the PDF.
        
        Args:
            pdf (PDF): The PDF object.
        """
        pdf.chapter_title("Equity Curve")
        
        # Create equity curve chart
        fig = self._create_equity_curve_chart()
        pdf.add_plot(fig, caption="Equity Curve")
        
        # Add drawdown chart
        fig = self._create_drawdown_chart()
        pdf.add_plot(fig, caption="Drawdown")
    
    def _add_trades_section(self, pdf: PDF, limit: Optional[int] = None) -> None:
        """
        Add trades section to the PDF.
        
        Args:
            pdf (PDF): The PDF object.
            limit (Optional[int], optional): Maximum number of trades to show. Defaults to None.
        """
        pdf.chapter_title("Trades")
        
        if not self.trade_log:
            pdf.body_text("No trades have been executed yet.")
            return
        
        # Sort trades by exit time (most recent first)
        sorted_trades = sorted(
            self.trade_log, 
            key=lambda x: x.get("exit_time", datetime.now()),
            reverse=True
        )
        
        # Limit number of trades if specified
        if limit:
            pdf.body_text(f"Showing the {limit} most recent trades. Total trades: {len(sorted_trades)}")
            sorted_trades = sorted_trades[:limit]
        
        # Create table data
        headers = ["ID", "Type", "Entry", "Exit", "P&L", "Pips", "Strategy"]
        data = []
        
        for trade in sorted_trades:
            trade_id = trade.get("id", "")
            
            # Format dates
            entry_time = trade.get("entry_time")
            if isinstance(entry_time, datetime):
                entry_time = entry_time.strftime("%m/%d %H:%M")
            elif isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00')).strftime("%m/%d %H:%M")
                except:
                    pass
            
            exit_time = trade.get("exit_time")
            if isinstance(exit_time, datetime):
                exit_time = exit_time.strftime("%m/%d %H:%M")
            elif isinstance(exit_time, str):
                try:
                    exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00')).strftime("%m/%d %H:%M")
                except:
                    pass
            
            data.append([
                trade_id,
                trade.get("type", "").upper(),
                entry_time,
                exit_time,
                f"${trade.get('pnl', 0):.2f}",
                f"{trade.get('pnl_pips', 0):.1f}",
                trade.get("strategy", "")
            ])
        
        # Add table
        pdf.add_table(headers, data, width=27)
    
    def _add_distribution_section(self, pdf: PDF) -> None:
        """
        Add distribution section to the PDF.
        
        Args:
            pdf (PDF): The PDF object.
        """
        pdf.chapter_title("Trade Distributions")
        
        # Create P&L distribution chart
        fig = self._create_pnl_distribution_chart()
        pdf.add_plot(fig, caption="P&L Distribution")
        
        # Create trade duration distribution chart
        fig = self._create_duration_distribution_chart()
        pdf.add_plot(fig, caption="Trade Duration Distribution")
    
    def _add_monthly_performance_section(self, pdf: PDF) -> None:
        """
        Add monthly performance section to the PDF.
        
        Args:
            pdf (PDF): The PDF object.
        """
        pdf.chapter_title("Monthly Performance")
        
        # Get monthly returns
        monthly_returns = self.metrics.calculate_monthly_returns()
        
        if not monthly_returns:
            pdf.body_text("No monthly data available yet.")
            return
        
        # Create monthly returns chart
        fig = self._create_monthly_returns_chart(monthly_returns)
        pdf.add_plot(fig, caption="Monthly Returns")
        
        # Create table data
        headers = ["Month", "P&L", "Return"]
        data = []
        
        # Sort months in descending order
        sorted_months = sorted(monthly_returns.keys(), reverse=True)
        
        for month in sorted_months:
            pnl = monthly_returns[month]
            monthly_return = pnl / INITIAL_CAPITAL * 100
            
            data.append([
                month,
                f"${pnl:.2f}",
                f"{monthly_return:.2f}%"
            ])
        
        # Add table
        pdf.add_table(headers, data, width=63)
    
    def _create_performance_charts(self) -> plt.Figure:
        """
        Create performance charts.
        
        Returns:
            plt.Figure: The figure with performance charts.
        """
        # Set up figure
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 2, figure=fig)
        
        # Style settings
        plt.style.use('dark_background')
        
        # Get metrics
        metrics = self.metrics.metrics
        total_trades = metrics.get("total_trades", 0)
        winning_trades = metrics.get("winning_trades", 0)
        losing_trades = metrics.get("losing_trades", 0)
        
        # Win/Loss Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        labels = ['Winning Trades', 'Losing Trades']
        sizes = [winning_trades, losing_trades]
        colors = ['#00FF00', '#FF0000']
        
        if total_trades > 0:
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, wedgeprops=dict(width=0.5, edgecolor='w'))
        else:
            ax1.text(0.5, 0.5, "No trades yet", ha='center', va='center')
        
        ax1.set_title('Win/Loss Ratio')
        
        # P&L by Strategy
        ax2 = fig.add_subplot(gs[0, 1])
        
        strategy_performance = metrics.get("strategy_performance", {})
        
        if strategy_performance:
            strategies = list(strategy_performance.keys())
            pnls = [strategy_performance[s]["pnl"] for s in strategies]
            
            bars = ax2.bar(strategies, pnls, color=self.colors['accent'])
            
            # Color bars based on P&L
            for i, p in enumerate(pnls):
                if p >= 0:
                    bars[i].set_color(self.colors['profit'])
                else:
                    bars[i].set_color(self.colors['loss'])
            
            ax2.set_title('P&L by Strategy')
            ax2.set_ylabel('P&L ($)')
            plt.xticks(rotation=45, ha='right')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "No strategy data yet", ha='center', va='center')
            ax2.set_title('P&L by Strategy')
        
        # Monthly Returns
        ax3 = fig.add_subplot(gs[1, :])
        
        monthly_returns = self.metrics.calculate_monthly_returns()
        
        if monthly_returns:
            months = list(monthly_returns.keys())
            returns = list(monthly_returns.values())
            
            # Sort by month
            months, returns = zip(*sorted(zip(months, returns)))
            
            bars = ax3.bar(months, returns, color=self.colors['accent'])
            
            # Color bars based on return
            for i, r in enumerate(returns):
                if r >= 0:
                    bars[i].set_color(self.colors['profit'])
                else:
                    bars[i].set_color(self.colors['loss'])
            
            ax3.set_title('Monthly Returns')
            ax3.set_ylabel('P&L ($)')
            plt.xticks(rotation=45, ha='right')
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax3.text(0.5, 0.5, "No monthly data yet", ha='center', va='center')
            ax3.set_title('Monthly Returns')
        
        plt.tight_layout()
        return fig
    
    def _create_strategy_comparison_chart(self) -> plt.Figure:
        """
        Create strategy comparison chart.
        
        Returns:
            plt.Figure: The figure with strategy comparison.
        """
        # Set up figure
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 2, figure=fig)
        
        # Style settings
        plt.style.use('dark_background')
        
        # Get metrics
        metrics = self.metrics.metrics
        strategy_performance = metrics.get("strategy_performance", {})
        
        if not strategy_performance:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No strategy data yet", ha='center', va='center')
            plt.tight_layout()
            return fig
        
        # Prepare data
        strategies = list(strategy_performance.keys())
        win_rates = [strategy_performance[s].get("win_rate", 0) for s in strategies]
        pnls = [strategy_performance[s].get("pnl", 0) for s in strategies]
        trade_counts = [strategy_performance[s].get("trades", 0) for s in strategies]
        
        # Win Rate Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(strategies, win_rates, color=self.colors['accent'])
        ax1.set_title('Win Rate by Strategy')
        ax1.set_ylabel('Win Rate (%)')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # P&L Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(strategies, pnls, color=self.colors['accent'])
        
        # Color bars based on P&L
        for i, p in enumerate(pnls):
            if p >= 0:
                bars[i].set_color(self.colors['profit'])
            else:
                bars[i].set_color(self.colors['loss'])
        
        ax2.set_title('P&L by Strategy')
        ax2.set_ylabel('P&L ($)')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Trade Count Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(strategies, trade_counts, color=self.colors['accent2'])
        ax3.set_title('Trade Count by Strategy')
        ax3.set_ylabel('Number of Trades')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Average P&L per Trade
        ax4 = fig.add_subplot(gs[1, 1])
        avg_pnls = [p / t if t > 0 else 0 for p, t in zip(pnls, trade_counts)]
        bars = ax4.bar(strategies, avg_pnls, color=self.colors['accent2'])
        
        # Color bars based on average P&L
        for i, p in enumerate(avg_pnls):
            if p >= 0:
                bars[i].set_color(self.colors['profit'])
            else:
                bars[i].set_color(self.colors['loss'])
        
        ax4.set_title('Average P&L per Trade')
        ax4.set_ylabel('P&L per Trade ($)')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def _create_equity_curve_chart(self) -> plt.Figure:
        """
        Create equity curve chart.
        
        Returns:
            plt.Figure: The figure with equity curve.
        """
        # Set up figure
        fig = plt.figure(figsize=(10, 6))
        
        # Style settings
        plt.style.use('dark_background')
        
        # Equity Curve
        plt.plot(self.equity_curve, color=self.colors['accent'], linewidth=2)
        
        # Add initial capital as horizontal line
        plt.axhline(y=INITIAL_CAPITAL, color='white', linestyle='--', alpha=0.5)
        
        # Set labels and title
        plt.title('Equity Curve')
        plt.ylabel('Capital ($)')
        plt.xlabel('Trades')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add markers for trades if we have trade log
        if self.trade_log:
            # Create a dummy index for x-axis
            x = list(range(len(self.equity_curve)))
            
            # Add markers for each trade
            for i, trade in enumerate(self.trade_log):
                pnl = trade.get("pnl", 0)
                if pnl > 0:
                    plt.plot(i+1, self.equity_curve[i+1], 'go', markersize=4)  # Green dot for winning trades
                elif pnl < 0:
                    plt.plot(i+1, self.equity_curve[i+1], 'ro', markersize=4)  # Red dot for losing trades
        
        plt.tight_layout()
        return fig
    
    def _create_drawdown_chart(self) -> plt.Figure:
        """
        Create drawdown chart.
        
        Returns:
            plt.Figure: The figure with drawdown.
        """
        # Set up figure
        fig = plt.figure(figsize=(10, 4))
        
        # Style settings
        plt.style.use('dark_background')
        
        # Calculate drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        
        # Plot drawdown
        plt.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        plt.plot(drawdown, color='red', linewidth=1)
        
        # Set labels and title
        plt.title('Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Trades')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Set y-axis limits
        plt.ylim(min(drawdown) * 1.1, 1)
        
        plt.tight_layout()
        return fig
    
    def _create_pnl_distribution_chart(self) -> plt.Figure:
        """
        Create P&L distribution chart.
        
        Returns:
            plt.Figure: The figure with P&L distribution.
        """
        # Set up figure
        fig = plt.figure(figsize=(10, 6))
        
        # Style settings
        plt.style.use('dark_background')
        
        if not self.trade_log:
            plt.text(0.5, 0.5, "No trades yet", ha='center', va='center')
            plt.title('P&L Distribution')
            plt.tight_layout()
            return fig
        
        # Extract P&L values
        pnl_values = [trade.get("pnl", 0) for trade in self.trade_log]
        
        # Create histogram
        n, bins, patches = plt.hist(pnl_values, bins=20, alpha=0.75, color=self.colors['accent'])
        
        # Color bins based on P&L
        for i in range(len(patches)):
            if bins[i] >= 0:
                patches[i].set_facecolor(self.colors['profit'])
            else:
                patches[i].set_facecolor(self.colors['loss'])
        
        # Add vertical line at zero
        plt.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        
        # Set labels and title
        plt.title('P&L Distribution')
        plt.ylabel('Frequency')
        plt.xlabel('P&L ($)')
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def _create_duration_distribution_chart(self) -> plt.Figure:
        """
        Create trade duration distribution chart.
        
        Returns:
            plt.Figure: The figure with trade duration distribution.
        """
        # Set up figure
        fig = plt.figure(figsize=(10, 6))
        
        # Style settings
        plt.style.use('dark_background')
        
        if not self.trade_log:
            plt.text(0.5, 0.5, "No trades yet", ha='center', va='center')
            plt.title('Trade Duration Distribution')
            plt.tight_layout()
            return fig
        
        # Calculate trade durations in hours
        durations = []
        
        for trade in self.trade_log:
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            
            if entry_time and exit_time:
                # Convert to datetime if string
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    except:
                        continue
                
                if isinstance(exit_time, str):
                    try:
                        exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                    except:
                        continue
                
                # Calculate duration in hours
                duration = (exit_time - entry_time).total_seconds() / 3600
                durations.append(duration)
        
        if not durations:
            plt.text(0.5, 0.5, "No duration data available", ha='center', va='center')
            plt.title('Trade Duration Distribution')
            plt.tight_layout()
            return fig
        
        # Create histogram
        plt.hist(durations, bins=20, alpha=0.75, color=self.colors['accent2'])
        
        # Set labels and title
        plt.title('Trade Duration Distribution')
        plt.ylabel('Frequency')
        plt.xlabel('Duration (hours)')
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def _create_monthly_returns_chart(self, monthly_returns: Dict[str, float]) -> plt.Figure:
        """
        Create monthly returns chart.
        
        Args:
            monthly_returns (Dict[str, float]): Monthly returns.
            
        Returns:
            plt.Figure: The figure with monthly returns.
        """
        # Set up figure
        fig = plt.figure(figsize=(10, 6))
        
        # Style settings
        plt.style.use('dark_background')
        
        if not monthly_returns:
            plt.text(0.5, 0.5, "No monthly data yet", ha='center', va='center')
            plt.title('Monthly Returns')
            plt.tight_layout()
            return fig
        
        # Sort months
        months = sorted(monthly_returns.keys())
        returns = [monthly_returns[m] for m in months]
        
        # Convert returns to percentages
        returns_pct = [r / INITIAL_CAPITAL * 100 for r in returns]
        
        # Create bar chart
        bars = plt.bar(months, returns_pct, color=self.colors['accent'])
        
        # Color bars based on return
        for i, r in enumerate(returns_pct):
            if r >= 0:
                bars[i].set_color(self.colors['profit'])
            else:
                bars[i].set_color(self.colors['loss'])
        
        # Set labels and title
        plt.title('Monthly Returns')
        plt.ylabel('Return (%)')
        plt.xlabel('Month')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def should_generate_report(self, frequency: str = PDF_REPORT_FREQUENCY) -> bool:
        """
        Check if a report should be generated based on frequency.
        
        Args:
            frequency (str, optional): Report frequency. 
                Defaults to PDF_REPORT_FREQUENCY from config.
                Options: "daily", "weekly", "monthly", "never".
                
        Returns:
            bool: True if a report should be generated, False otherwise.
        """
        if not GENERATE_PDF_REPORTS:
            return False
        
        if frequency == "never":
            return False
        
        # Get all report files
        report_files = [f for f in os.listdir(self.output_dir) if f.startswith("report_") and f.endswith(".pdf")]
        
        if not report_files:
            # No reports yet, generate one
            return True
        
        # Get the latest report timestamp
        latest_file = max(report_files, key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)))
        latest_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(self.output_dir, latest_file)))
        
        # Current time
        now = datetime.now()
        
        # Check frequency
        if frequency == "daily":
            # Generate if last report was not today
            return latest_time.date() < now.date()
        elif frequency == "weekly":
            # Generate if last report was more than 7 days ago
            return (now - latest_time).days >= 7
        elif frequency == "monthly":
            # Generate if last report was in a different month or more than 30 days ago
            return (now.year != latest_time.year or now.month != latest_time.month or 
                    (now - latest_time).days >= 30)
        
        return False