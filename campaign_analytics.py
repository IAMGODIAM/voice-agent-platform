"""
Campaign Analytics Dashboard — VibeVoice Outbound Call Center
Real-time monitoring, reporting, and analytics.
"""

import json
import os
import time
from datetime import datetime, timedelta
from collections import Counter
from dataclasses import dataclass

RESULTS_DIR = os.getenv("CALL_RESULTS_DIR", "/home/user/hermes-workspace/voice-agent-platform/call_results")


@dataclass
class CampaignMetrics:
    """Aggregate metrics for a campaign."""
    campaign_id: str
    campaign_name: str
    total_calls: int = 0
    completed: int = 0
    failed: int = 0
    no_answer: int = 0
    voicemail_drops: int = 0
    qualified: int = 0
    callbacks: int = 0
    not_interested: int = 0
    dnc: int = 0
    total_talk_time: float = 0.0
    avg_interest_score: float = 0.0
    conversion_rate: float = 0.0
    cost_per_lead: float = 0.0
    cost_per_qualified: float = 0.0
    roi_estimate: float = 0.0


class CampaignAnalytics:
    """Analytics engine for outbound campaigns."""
    
    # Default cost assumptions
    COST_PER_MINUTE = 0.013  # Twilio
    AVG_DEAL_VALUE = 500.0  # Expected revenue per qualified lead
    
    def load_results(self, results_file: str) -> dict:
        """Load campaign results from JSON."""
        try:
            with open(results_file) as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    
    def load_all_results(self) -> list[dict]:
        """Load all campaign results."""
        all_results = []
        results_path = Path(RESULTS_DIR)
        
        if not results_path.exists():
            return all_results
        
        for f in results_path.glob("campaign_*.json"):
            try:
                with open(f) as fh:
                    all_results.append(json.load(fh))
            except Exception:
                pass
        
        return all_results
    
    def compute_metrics(self, campaign_data: dict) -> CampaignMetrics:
        """Compute aggregate metrics from raw campaign data."""
        summary = campaign_data.get("summary", {})
        results = campaign_data.get("results", [])
        campaign = campaign_data.get("campaign", {})
        
        metrics = CampaignMetrics(
            campaign_id=campaign.get("id", ""),
            campaign_name=campaign.get("name", ""),
            total_calls=summary.get("total_calls", 0),
            completed=summary.get("completed", 0),
            failed=summary.get("failed", 0),
            qualified=summary.get("qualified", 0),
            callbacks=summary.get("callbacks", 0),
        )
        
        # Disposition breakdown
        dispositions = Counter(r.get("disposition", "unknown") for r in results if r.get("disposition"))
        metrics.not_interested = dispositions.get("not_interested", 0)
        metrics.dnc = dispositions.get("dnc", 0)
        metrics.no_answer = dispositions.get("no_answer", 0)
        metrics.voicemail_drops = dispositions.get("voicemail_left", 0)
        
        # Time metrics
        durations = [r.get("duration", 0) for r in results if r.get("duration", 0) > 0]
        metrics.total_talk_time = sum(durations)
        
        # Interest
        scores = [r.get("interest_score", 0) for r in results if r.get("interest_score")]
        metrics.avg_interest_score = sum(scores) / max(len(scores), 1)
        
        # Financials
        call_cost = (metrics.total_talk_time / 60) * self.COST_PER_MINUTE
        metrics.cost_per_lead = call_cost / max(metrics.total_calls, 1)
        metrics.cost_per_qualified = call_cost / max(metrics.qualified, 1)
        
        expected_revenue = metrics.qualified * self.AVG_DEAL_VALUE * 0.1  # 10% close rate
        metrics.roi_estimate = (expected_revenue - call_cost) / max(call_cost, 0.01)
        
        # Conversion
        metrics.conversion_rate = metrics.qualified / max(metrics.completed, 1)
        
        return metrics
    
    def generate_report(self, metrics: CampaignMetrics) -> str:
        """Generate human-readable campaign report."""
        avg_talk = metrics.total_talk_time / max(metrics.completed, 1)
        total_cost = (metrics.total_talk_time / 60) * self.COST_PER_MINUTE
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           CAMPAIGN REPORT: {metrics.campaign_name[:30]:30s} ║
╠══════════════════════════════════════════════════════════╣
║ CALL VOLUME                                               ║
║   Total Calls:     {metrics.total_calls:>6}                              ║
║   Completed:       {metrics.completed:>6}  ({100*metrics.completed/max(metrics.total_calls,1):.1f}%)                     ║
║   Failed:          {metrics.failed:>6}                              ║
║   No Answer:       {metrics.no_answer:>6}                              ║
║   Voicemail Drops: {metrics.voicemail_drops:>6}                              ║
╠══════════════════════════════════════════════════════════╣
║ DISPOSITIONS                                              ║
║   Qualified:       {metrics.qualified:>6}  ← HOT LEADS                    ║
║   Callbacks:       {metrics.callbacks:>6}  ← FOLLOW UP                   ║
║   Not Interested:  {metrics.not_interested:>6}                              ║
║   DNC:             {metrics.dnc:>6}  ← REMOVE FROM LIST           ║
╠══════════════════════════════════════════════════════════╣
║ PERFORMANCE                                               ║
║   Conversion Rate:  {100*metrics.conversion_rate:>5.1f}%                            ║
║   Avg Interest:     {100*metrics.avg_interest_score:>5.1f}%                            ║
║   Avg Talk Time:    {avg_talk:>5.1f}s                             ║
╠══════════════════════════════════════════════════════════╣
║ FINANCIALS                                                ║
║   Total Cost:       ${total_cost:>7.2f}                           ║
║   Cost/Lead:        ${metrics.cost_per_lead:>7.4f}                           ║
║   Cost/Qualified:   ${metrics.cost_per_qualified:>7.2f}                           ║
║   Est. ROI:         {metrics.roi_estimate:>6.1f}x                            ║
╚══════════════════════════════════════════════════════════╝

RECOMMENDATIONS:
"""
        if metrics.conversion_rate < 0.05:
            report += "  ⚠ Low conversion — review opening script and targeting\n"
        if metrics.no_answer > metrics.total_calls * 0.3:
            report += "  ⚠ High no-answer rate — try different call times\n"
        if metrics.avg_interest_score < 0.3:
            report += "  ⚠ Low interest scores — personalize scripts more\n"
        if metrics.qualified > 0 and metrics.cost_per_qualified < 5.0:
            report += "  ✅ Good cost/qualified — scale this campaign\n"
        if metrics.callbacks > metrics.qualified:
            report += "  📞 Schedule callback follow-ups within 2 hours\n"
        
        return report
    
    def generate_lead_hotlist(self, campaign_data: dict) -> list[dict]:
        """Extract hot leads (qualified + high interest) for immediate follow-up."""
        results = campaign_data.get("results", [])
        hot = []
        
        for r in results:
            score = r.get("interest_score", 0)
            disp = r.get("disposition", "")
            
            if disp == "lead_qualified" or score >= 0.6:
                hot.append({
                    "call_id": r.get("call_id"),
                    "lead_id": r.get("lead_id"),
                    "disposition": disp,
                    "interest_score": score,
                    "duration": r.get("duration", 0),
                })
        
        hot.sort(key=lambda x: x["interest_score"], reverse=True)
        return hot
    
    def export_hotlist_csv(self, hotlist: list[dict], output_file: str = None) -> str:
        """Export hot leads to CSV for CRM import."""
        import csv
        from typing import Optional
        
        if output_file is None:
            output_file = os.path.join(RESULTS_DIR, f"hotlist_{int(time.time())}.csv")
        
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["call_id", "lead_id", "disposition", "interest_score", "duration"])
            writer.writeheader()
            writer.writerows(hotlist)
        
        return output_file


# ── ASCII Real-time Dashboard ─────────────────────────────────────────────

def render_dashboard(active_calls: list, metrics: CampaignMetrics) -> str:
    """Render real-time ASCII dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Active calls bar
    active_bar = "█" * len(active_calls) + "░" * (10 - min(len(active_calls), 10))
    
    # Disposition pie (ASCII)
    total = max(metrics.completed, 1)
    qual_pct = int(20 * metrics.qualified / total)
    cb_pct = int(20 * metrics.callbacks / total)
    ni_pct = int(20 * metrics.not_interested / total)
    
    dashboard = f"""
┌─────────────────────────────────────────────────┐
│  🎯 VIBVOICE OUTBOUND CENTER — {now}  │
├─────────────────────────────────────────────────┤
│  ACTIVE CALLS: [{active_bar}] {len(active_calls):>2}/10           │
├─────────────────────────────────────────────────┤
│  TODAY'S RESULTS          │  DISPOSITION MIX     │
│  Calls:    {metrics.total_calls:>6}            │  🟢 Qualified: {metrics.qualified:>4}   │
│  Completed:{metrics.completed:>6}            │  🟡 Callback:  {metrics.callbacks:>4}   │
│  Failed:   {metrics.failed:>6}            │  🔴 Not Inter: {metrics.not_interested:>4}   │
│  Talk Time:{metrics.total_talk_time/60:>5.1f}min        │  ⚫ DNC:       {metrics.dnc:>4}   │
├───────────────────────────┼─────────────────────┤
│  CONVERSION: {100*metrics.conversion_rate:>5.1f}%        │  COST/QUAL: ${metrics.cost_per_qualified:>6.2f}  │
│  AVG INTEREST: {100*metrics.avg_interest_score:>4.1f}%      │  ROI:     {metrics.roi_estimate:>5.1f}x   │
└─────────────────────────────────────────────────┘
"""
    return dashboard


from pathlib import Path

if __name__ == "__main__":
    analytics = CampaignAnalytics()
    
    # Load and report on all campaigns
    all_campaigns = analytics.load_all_results()
    
    if not all_campaigns:
        print("No campaign results found.")
        print(f"Place campaign result files in: {RESULTS_DIR}")
    else:
        for campaign_data in all_campaigns:
            metrics = analytics.compute_metrics(campaign_data)
            report = analytics.generate_report(metrics)
            print(report)
            
            hotlist = analytics.generate_lead_hotlist(campaign_data)
            if hotlist:
                print(f"  🔥 HOT LEADS: {len(hotlist)}")
                for lead in hotlist[:5]:
                    print(f"     {lead['lead_id']} — score: {lead['interest_score']:.1%} — {lead['disposition']}")
