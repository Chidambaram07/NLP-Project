import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search, Zap, TrendingUp, MessageSquare, ChevronDown,
  Sparkles, BarChart3, Users, DollarSign, ArrowRight, Loader2,
  Clock, Activity, Lightbulb, CheckCircle2, Target, Flame,
  Globe, Shield, Layers, Sun, Moon, ChevronRight
} from "lucide-react";

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const API_BASE = "http://127.0.0.1:8000/api/v1";

// ─── MOCK DATA ────────────────────────────────────────────────────────────────
const MOCK_PAIN_POINTS = [
  { title: "Can't get hired as a fresher despite 200+ applications", category: "Career", score: 892, num_comments: 347, content: "I've applied to so many companies but get ghosted every time. No response, no feedback. It's been 8 months." },
  { title: "Mental health resources in India are inaccessible and unaffordable", category: "Health", score: 654, num_comments: 289, content: "Therapists charge 2000+ per session. Most people can't afford this. There's a massive gap." },
  { title: "Rent in Bangalore is insane — landlords want 6 months deposit upfront", category: "Housing", score: 543, num_comments: 201, content: "Moving to Bangalore for work but the rental market is exploitative. No transparency whatsoever." },
  { title: "Learning to code alone is impossible without proper mentorship", category: "Education", score: 478, num_comments: 176, content: "YouTube tutorials get you so far. Without someone to review your code and guide you, you hit a wall." },
  { title: "Freelance payment delays are killing small businesses", category: "Business", score: 391, num_comments: 143, content: "Clients in India pay 60-90 days late. No legal recourse for freelancers. Platform fees eat the rest." },
];

const MOCK_SOLUTION = {
  problem_statement: "Fresh graduates face a broken hiring system with zero feedback loops, making it impossible to improve or understand rejection reasons.",
  solutions: [
    { idea: "FreshTrack — Application Status & Feedback Platform", how_it_works: "A SaaS tool for companies to send structured feedback after rejections. Freshers get actionable insights; companies get employer brand boost.", why_it_works: "Companies want better employer brand. Candidates desperately want feedback. This bridges both." },
    { idea: "PeerHire — Community-Verified Portfolios", how_it_works: "A platform where senior developers vouch for junior candidates through code reviews and mock interviews, replacing traditional resumes.", why_it_works: "Trust-based hiring removes credential gatekeeping. Companies get pre-vetted talent." },
    { idea: "SkillMap — Micro-Internship Marketplace", how_it_works: "3-7 day paid micro-internships where freshers prove skills on real tasks. Companies hire the best performers full-time.", why_it_works: "Lower risk for companies, real experience for freshers. Solves the 'experience paradox' at root." },
  ],
  target_audience: "CS/IT freshers (18-24), tier-2/3 college graduates, career switchers with no formal experience",
  monetization: "B2B SaaS subscription for companies (₹5K-₹50K/month based on hiring volume) + premium profiles for candidates (₹299/month)",
  market_size: "large",
  difficulty: "medium"
};

// ─── CATEGORY CONFIG ──────────────────────────────────────────────────────────
const CAT_CONFIG = {
  Career:        { light: { bg: "#EEF2FF", text: "#3730A3", dot: "#6366F1", bar: "#818CF8" }, dark: { bg: "#1e1b4b", text: "#a5b4fc", dot: "#6366F1", bar: "#6366F1" } },
  Health:        { light: { bg: "#ECFDF5", text: "#065F46", dot: "#10B981", bar: "#34D399" }, dark: { bg: "#052e16", text: "#6ee7b7", dot: "#10B981", bar: "#10B981" } },
  Housing:       { light: { bg: "#FFF1F2", text: "#9F1239", dot: "#F43F5E", bar: "#FB7185" }, dark: { bg: "#4c0519", text: "#fda4af", dot: "#F43F5E", bar: "#F43F5E" } },
  Education:     { light: { bg: "#FFFBEB", text: "#92400E", dot: "#F59E0B", bar: "#FCD34D" }, dark: { bg: "#451a03", text: "#fde68a", dot: "#F59E0B", bar: "#F59E0B" } },
  Finance:       { light: { bg: "#EFF6FF", text: "#1E40AF", dot: "#3B82F6", bar: "#60A5FA" }, dark: { bg: "#172554", text: "#93c5fd", dot: "#3B82F6", bar: "#3B82F6" } },
  Technology:    { light: { bg: "#F5F3FF", text: "#5B21B6", dot: "#8B5CF6", bar: "#A78BFA" }, dark: { bg: "#2e1065", text: "#c4b5fd", dot: "#8B5CF6", bar: "#8B5CF6" } },
  Business:      { light: { bg: "#FFF7ED", text: "#9A3412", dot: "#F97316", bar: "#FB923C" }, dark: { bg: "#431407", text: "#fdba74", dot: "#F97316", bar: "#F97316" } },
  Lifestyle:     { light: { bg: "#ECFEFF", text: "#155E75", dot: "#06B6D4", bar: "#22D3EE" }, dark: { bg: "#083344", text: "#67e8f9", dot: "#06B6D4", bar: "#06B6D4" } },
  Relationships: { light: { bg: "#FDF2F8", text: "#86198F", dot: "#D946EF", bar: "#E879F9" }, dark: { bg: "#4a044e", text: "#f0abfc", dot: "#D946EF", bar: "#D946EF" } },
  General:       { light: { bg: "#F8FAFC", text: "#475569", dot: "#94A3B8", bar: "#CBD5E1" }, dark: { bg: "#1e293b", text: "#94a3b8", dot: "#64748B", bar: "#64748B" } },
};
const getCat = (cat, dark) => (CAT_CONFIG[cat] || CAT_CONFIG.General)[dark ? "dark" : "light"];

// ─── THEME TOKENS ─────────────────────────────────────────────────────────────
const T = {
  light: {
    page:        "#F8F7F4",
    header:      "rgba(248,247,244,0.95)",
    panel:       "#FFFFFF",
    card:        "#FFFFFF",
    cardHover:   "#FAFAF9",
    inner:       "#F8F7F4",
    border:      "#E2E0DC",
    borderMid:   "#D1CFC9",
    borderStrong:"#B8B5AE",
    text:        "#1A1916",
    textSec:     "#605E58",
    textTer:     "#9C9890",
    textPlaceholder: "#B8B5AE",
    shadow:      "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)",
    shadowMd:    "0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04)",
    shadowLg:    "0 8px 32px rgba(0,0,0,0.1), 0 2px 8px rgba(0,0,0,0.05)",
    accent:      "#1A1916",
    accentText:  "#FFFFFF",
    inputBg:     "#FFFFFF",
  },
  dark: {
    page:        "#0F0E0C",
    header:      "rgba(15,14,12,0.95)",
    panel:       "#1C1B18",
    card:        "#161513",
    cardHover:   "#1E1D1A",
    inner:       "#111009",
    border:      "#2A2926",
    borderMid:   "#333230",
    borderStrong:"#4A4845",
    text:        "#E8E6E1",
    textSec:     "#A09D96",
    textTer:     "#6B6862",
    textPlaceholder: "#4A4845",
    shadow:      "0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)",
    shadowMd:    "0 4px 16px rgba(0,0,0,0.4), 0 1px 4px rgba(0,0,0,0.3)",
    shadowLg:    "0 8px 40px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.3)",
    accent:      "#E8E6E1",
    accentText:  "#0F0E0C",
    inputBg:     "#1C1B18",
  }
};

// ─── DIFFICULTY / MARKET ──────────────────────────────────────────────────────
const getDiff = (d, dark) => ({
  easy:   { label: "Low risk",    light: { bg: "#ECFDF5", text: "#065F46", dot: "#10B981" }, dark: { bg: "#052e16", text: "#6ee7b7", dot: "#10B981" } },
  medium: { label: "Medium risk", light: { bg: "#FFFBEB", text: "#92400E", dot: "#F59E0B" }, dark: { bg: "#451a03", text: "#fde68a", dot: "#F59E0B" } },
  hard:   { label: "High risk",   light: { bg: "#FFF1F2", text: "#9F1239", dot: "#F43F5E" }, dark: { bg: "#4c0519", text: "#fda4af", dot: "#F43F5E" } },
}[d] || { label: "Medium risk", light: { bg: "#FFFBEB", text: "#92400E", dot: "#F59E0B" }, dark: { bg: "#451a03", text: "#fde68a", dot: "#F59E0B" } })[dark ? "dark" : "light"];

const getMkt = (m, dark) => ({
  small:  { label: "Niche",   Icon: Globe      },
  medium: { label: "Growth",  Icon: TrendingUp },
  large:  { label: "Massive", Icon: Flame      },
}[m] || { label: "Growth", Icon: TrendingUp });

// ─── SMALL COMPONENTS ────────────────────────────────────────────────────────

function Tag({ children, bg, color, style = {} }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "3px 8px", borderRadius: 5,
      fontSize: 10.5, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase",
      background: bg, color, whiteSpace: "nowrap", flexShrink: 0, ...style,
    }}>
      {children}
    </span>
  );
}

function Bar({ color, width }) {
  return (
    <div style={{ height: 3, borderRadius: 99, background: "rgba(128,128,128,0.12)", overflow: "hidden" }}>
      <motion.div
        initial={{ width: 0 }} animate={{ width }}
        transition={{ duration: 0.8, delay: 0.1, ease: [0.4, 0, 0.2, 1] }}
        style={{ height: "100%", borderRadius: 99, background: color }}
      />
    </div>
  );
}

function Spinner({ t }) {
  return (
    <div style={{ padding: "52px 24px", display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
      <div style={{ width: 32, height: 32, borderRadius: "50%", border: `2px solid ${t.border}`, borderTopColor: t.text, animation: "spin 0.9s linear infinite" }} />
      <p style={{ fontSize: 13, color: t.textSec, margin: 0, fontFamily: "'DM Mono', monospace" }}>
        analysing opportunity…
      </p>
    </div>
  );
}

function SkeletonCard({ t }) {
  const p = (w, h, mb = 10) => (
    <div style={{ width: w, height: h, borderRadius: 4, background: t.border, marginBottom: mb, animation: "pulse 1.8s ease-in-out infinite alternate" }} />
  );
  return (
    <div style={{ borderRadius: 10, border: `1px solid ${t.border}`, background: t.card, padding: "16px 18px" }}>
      {p("28%", 16)}{p("100%", 12)}{p("72%", 12)}{p("100%", 3)}{p("45%", 11, 0)}
    </div>
  );
}

// ─── PAIN CARD ────────────────────────────────────────────────────────────────
function PainCard({ point, onAnalyze, isLoading, isAnalyzed, isActive, dark }) {
  const t = T[dark ? "dark" : "light"];
  const c = getCat(point.category, dark);
  const [hov, setHov] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.22, ease: [0.25, 0.46, 0.45, 0.94] }}
      whileHover={{ y: -1 }}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        position: "relative",
        borderRadius: 10,
        border: `1px solid ${isActive ? t.borderMid : hov ? t.border : t.border}`,
        background: isActive ? t.cardHover : t.card,
        boxShadow: isActive ? t.shadowMd : hov ? t.shadow : "none",
        padding: "14px 16px",
        cursor: "pointer",
        transition: "all 0.18s ease",
        overflow: "hidden",
      }}
    >
      {/* Left accent bar — only active */}
      {isActive && (
        <div style={{
          position: "absolute", left: 0, top: 0, bottom: 0, width: 2.5,
          background: c.dot, borderRadius: "10px 0 0 10px",
        }} />
      )}

      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 8, marginBottom: 10 }}>
        <Tag bg={c.bg} color={c.text}>
          <span style={{ width: 4, height: 4, borderRadius: "50%", background: c.dot }} />
          {point.category}
        </Tag>
        {isAnalyzed && (
          <Tag bg={dark ? "#052e16" : "#ECFDF5"} color={dark ? "#6ee7b7" : "#065F46"}>
            <CheckCircle2 size={8} strokeWidth={2.5} />done
          </Tag>
        )}
      </div>

      <p style={{
        margin: "0 0 12px",
        fontSize: 13,
        fontWeight: isActive ? 500 : 400,
        lineHeight: 1.55,
        color: isActive ? t.text : hov ? t.textSec : t.textSec,
        transition: "color 0.15s",
        fontFamily: "'DM Sans', sans-serif",
      }}>
        {point.title}
      </p>

      <div style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
          <span style={{ fontSize: 10, color: t.textTer, display: "flex", alignItems: "center", gap: 3 }}>
            <Activity size={8} />demand signal
          </span>
          <span style={{ fontSize: 10, color: t.textTer, fontVariantNumeric: "tabular-nums", fontFamily: "'DM Mono', monospace" }}>
            {point.score?.toLocaleString()}
          </span>
        </div>
        <Bar color={c.bar} width={`${Math.min((point.score / 1000) * 100, 100)}%`} />
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: 10, fontSize: 11, color: t.textTer }}>
          <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <MessageSquare size={9} />{point.num_comments?.toLocaleString()}
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <TrendingUp size={9} />{point.score?.toLocaleString()}
          </span>
        </div>
        <AnalyzeButton
          isLoading={isLoading} isAnalyzed={isAnalyzed}
          dark={dark} t={t} dot={c.dot} bg={c.bg} textColor={c.text}
          onAnalyze={(e) => { e.stopPropagation(); onAnalyze(point); }}
        />
      </div>
    </motion.div>
  );
}

function AnalyzeButton({ isLoading, isAnalyzed, dark, t, dot, bg, textColor, onAnalyze }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      onClick={onAnalyze}
      disabled={isLoading}
      style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        padding: "5px 11px", borderRadius: 6,
        fontSize: 11, fontWeight: 500,
        cursor: isLoading ? "wait" : "pointer",
        fontFamily: "'DM Sans', sans-serif",
        transition: "all 0.15s ease",
        border: "none", outline: "none",
        ...(isAnalyzed
          ? {
            background: hov ? t.inner : "transparent",
            color: t.textSec,
            border: `1px solid ${t.border}`,
          }
          : {
            background: hov ? t.text : dot,
            color: "#fff",
            border: `1px solid ${dot}`,
          }
        ),
        opacity: isLoading ? 0.65 : 1,
      }}
    >
      {isLoading ? <><Loader2 size={9} style={{ animation: "spin 0.8s linear infinite" }} />Working…</>
        : isAnalyzed ? <><Sparkles size={9} />View breakdown</>
          : <><Zap size={9} />Find opportunity</>}
    </button>
  );
}

// ─── SOLUTION PANEL ───────────────────────────────────────────────────────────
function SolutionPanel({ data, painPoint, dark }) {
  const t = T[dark ? "dark" : "light"];
  const a = data.analysis || {};
  const [expanded, setExpanded] = useState(0);
  const c = getCat(data.category || painPoint?.category, dark);

  if (!a.solutions) {
    return (
      <div style={{ borderRadius: 12, border: `1px solid ${t.border}`, background: t.panel, padding: 20 }}>
        <pre style={{ fontSize: 11, color: t.textTer, whiteSpace: "pre-wrap", margin: 0, fontFamily: "'DM Mono', monospace" }}>
          {JSON.stringify(a, null, 2)}
        </pre>
      </div>
    );
  }

  const diff = getDiff(a.difficulty, dark);
  const mkt = getMkt(a.market_size, dark);
  const MktIcon = mkt.Icon;

  return (
    <motion.div
      key={data.pain_point}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 6 }}
      transition={{ duration: 0.28, ease: [0.25, 0.46, 0.45, 0.94] }}
      style={{
        borderRadius: 12,
        border: `1px solid ${t.border}`,
        background: t.panel,
        overflow: "hidden",
        boxShadow: t.shadowLg,
      }}
    >
      {/* Top accent — thin stripe using category color */}
      <div style={{ height: 3, background: c.dot }} />

      {/* Header */}
      <div style={{ padding: "20px 22px 18px", borderBottom: `1px solid ${t.border}` }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10, marginBottom: 14 }}>
          <div>
            <p style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: t.textTer, margin: "0 0 3px", fontFamily: "'DM Mono', monospace" }}>
              Opportunity breakdown
            </p>
            <p style={{ fontSize: 13, color: t.textSec, margin: 0, lineHeight: 1.5 }}>
              {painPoint?.title}
            </p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 5, alignItems: "flex-end", flexShrink: 0 }}>
            <Tag bg={diff.bg} color={diff.text}>
              <span style={{ width: 4, height: 4, borderRadius: "50%", background: diff.dot }} />{diff.label}
            </Tag>
            <Tag bg={dark ? "#1e293b" : "#F8FAFC"} color={dark ? "#94a3b8" : "#475569"}>
              <MktIcon size={9} />{mkt.label} market
            </Tag>
          </div>
        </div>

        {/* Problem statement */}
        <div style={{ background: t.inner, border: `1px solid ${t.border}`, borderRadius: 8, padding: "12px 14px" }}>
          <p style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: t.textTer, margin: "0 0 6px", fontFamily: "'DM Mono', monospace" }}>
            Core problem
          </p>
          <p style={{ margin: 0, fontSize: 12.5, color: t.textSec, lineHeight: 1.65 }}>
            {a.problem_statement}
          </p>
        </div>
      </div>

      {/* Solutions */}
      <div style={{ padding: "16px 22px 14px" }}>
        <p style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: t.textTer, margin: "0 0 10px", fontFamily: "'DM Mono', monospace", display: "flex", alignItems: "center", gap: 5 }}>
          <Lightbulb size={9} />3 ways to build on this
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {a.solutions.map((sol, i) => (
            <SolutionRow key={i} sol={sol} index={i} isExp={expanded === i} c={c} t={t} dark={dark}
              onToggle={() => setExpanded(expanded === i ? -1 : i)} />
          ))}
        </div>
      </div>

      {/* Footer grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, padding: "0 22px 20px" }}>
        {[
          { Icon: Users, label: "Who pays", val: a.target_audience, iconColor: c.dot, iconBg: c.bg },
          { Icon: DollarSign, label: "Revenue model", val: a.monetization, iconColor: dark ? "#6ee7b7" : "#065F46", iconBg: dark ? "#052e16" : "#ECFDF5" },
        ].map(({ Icon, label, val, iconColor, iconBg }) => (
          <div key={label} style={{ background: t.inner, border: `1px solid ${t.border}`, borderRadius: 8, padding: "11px 13px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 7 }}>
              <div style={{ width: 20, height: 20, borderRadius: 5, background: iconBg, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon size={9} style={{ color: iconColor }} />
              </div>
              <p style={{ margin: 0, fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: t.textTer, fontFamily: "'DM Mono', monospace" }}>{label}</p>
            </div>
            <p style={{ margin: 0, fontSize: 11.5, color: t.textSec, lineHeight: 1.6 }}>{val}</p>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function SolutionRow({ sol, index, isExp, c, t, dark, onToggle }) {
  const [hov, setHov] = useState(false);
  return (
    <motion.div
      layout
      onClick={onToggle}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        borderRadius: 7,
        border: `1px solid ${isExp ? t.borderMid : hov ? t.border : "transparent"}`,
        background: isExp ? t.inner : hov ? t.inner : "transparent",
        cursor: "pointer", overflow: "hidden",
        transition: "border-color 0.15s, background 0.15s",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "9px 12px" }}>
        <div style={{
          width: 20, height: 20, borderRadius: 5, flexShrink: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 10, fontWeight: 700, fontFamily: "'DM Mono', monospace",
          background: isExp ? c.dot : t.inner,
          color: isExp ? "#fff" : t.textTer,
          border: `1px solid ${isExp ? c.dot : t.border}`,
          transition: "all 0.15s",
        }}>
          {index + 1}
        </div>
        <span style={{
          flex: 1, fontSize: 12.5, fontWeight: isExp ? 500 : 400,
          color: isExp ? t.text : hov ? t.textSec : t.textSec,
          transition: "color 0.15s", lineHeight: 1.45,
        }}>
          {sol.idea}
        </span>
        <motion.div animate={{ rotate: isExp ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown size={11} style={{ color: t.textTer }} />
        </motion.div>
      </div>

      <AnimatePresence>
        {isExp && (
          <motion.div
            initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.22 }}
            style={{ overflow: "hidden" }}
          >
            <div style={{ padding: "0 12px 12px", display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ borderTop: `1px solid ${t.border}`, paddingTop: 10 }}>
                <p style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: c.text, margin: "0 0 5px", fontFamily: "'DM Mono', monospace" }}>
                  How it works
                </p>
                <p style={{ margin: 0, fontSize: 12, color: t.textSec, lineHeight: 1.7 }}>{sol.how_it_works}</p>
              </div>
              <div>
                <p style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", color: t.textTer, margin: "0 0 5px", fontFamily: "'DM Mono', monospace" }}>
                  The edge
                </p>
                <p style={{ margin: 0, fontSize: 12, color: t.textSec, lineHeight: 1.7 }}>{sol.why_it_works}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ─── EMPTY STATE ─────────────────────────────────────────────────────────────
function EmptyState({ t }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      style={{
        borderRadius: 12, border: `1px dashed ${t.border}`,
        background: t.panel, padding: "52px 32px 48px",
        display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center",
      }}
    >
      <div style={{
        width: 40, height: 40, borderRadius: 10,
        background: t.inner, border: `1px solid ${t.border}`,
        display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 16,
      }}>
        <Target size={18} style={{ color: t.textTer }} />
      </div>
      <h3 style={{ margin: "0 0 8px", fontSize: 15, fontWeight: 500, color: t.text, fontFamily: "'DM Sans', sans-serif", letterSpacing: "-0.01em" }}>
        Pick a problem, find a business
      </h3>
      <p style={{ margin: "0 0 24px", fontSize: 12.5, color: t.textSec, lineHeight: 1.7, maxWidth: 240 }}>
        Every item on the left has real founders earning from it. Hit <strong style={{ fontWeight: 500, color: t.text }}>Find opportunity</strong> to see how.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, width: "100%", maxWidth: 260 }}>
        {[
          { Icon: Target, label: "Root cause" },
          { Icon: Lightbulb, label: "3 startup ideas" },
          { Icon: BarChart3, label: "Market size" },
          { Icon: DollarSign, label: "Revenue model" },
        ].map(({ Icon, label }) => (
          <div key={label} style={{
            display: "flex", alignItems: "center", gap: 6,
            background: t.inner, border: `1px solid ${t.border}`,
            borderRadius: 7, padding: "7px 10px",
          }}>
            <Icon size={11} style={{ color: t.textTer }} strokeWidth={1.5} />
            <span style={{ fontSize: 11, color: t.textSec }}>{label}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// ─── FILTER PILL ─────────────────────────────────────────────────────────────
function FilterPill({ cat, c, active, isAll, onClick, dark }) {
  const t = T[dark ? "dark" : "light"];
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: "inline-flex", alignItems: "center", gap: 4,
        padding: "4px 10px", borderRadius: 6,
        fontSize: 11.5, fontWeight: active ? 500 : 400,
        cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
        transition: "all 0.15s ease", outline: "none",
        ...(active
          ? isAll
            ? { background: t.text, color: t.page, border: `1px solid ${t.text}` }
            : { background: c.bg, color: c.text, border: `1px solid ${c.dot}20` }
          : { background: "transparent", color: hov ? t.textSec : t.textTer, border: `1px solid ${hov ? t.border : "transparent"}` }
        ),
      }}
    >
      {active && !isAll && <span style={{ width: 4, height: 4, borderRadius: "50%", background: c.dot }} />}
      {cat}
    </button>
  );
}

// ─── HISTORY ITEM ─────────────────────────────────────────────────────────────
function HistoryItem({ histKey, isActive, allPainPoints, onClick, dark }) {
  const t = T[dark ? "dark" : "light"];
  const pp = allPainPoints.find(p => p.title === histKey);
  const c = getCat(pp?.category, dark);
  const [hov, setHov] = useState(false);
  return (
    <button
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      onClick={onClick}
      style={{
        display: "flex", alignItems: "center", gap: 7,
        padding: "6px 9px", borderRadius: 6, textAlign: "left",
        cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
        fontSize: 11.5,
        background: isActive ? t.inner : hov ? t.inner : "transparent",
        border: `1px solid ${isActive ? t.border : "transparent"}`,
        color: isActive ? t.text : t.textSec,
        transition: "all 0.15s ease", outline: "none",
      }}
    >
      <span style={{ width: 5, height: 5, borderRadius: "50%", flexShrink: 0, background: isActive ? c.dot : t.borderStrong }} />
      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {histKey.length > 52 ? histKey.slice(0, 52) + "…" : histKey}
      </span>
      {isActive && <ChevronRight size={10} style={{ color: t.textTer, flexShrink: 0 }} />}
    </button>
  );
}

// ─── STAT BOX ─────────────────────────────────────────────────────────────────
function StatBox({ label, value, t }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "flex-end",
      padding: "4px 12px",
      borderLeft: `1px solid ${t.border}`,
    }}>
      <span style={{ fontSize: 15, fontWeight: 500, color: t.text, fontFamily: "'DM Mono', monospace", lineHeight: 1.1 }}>
        {value}
      </span>
      <span style={{ fontSize: 9.5, fontWeight: 500, color: t.textTer, textTransform: "uppercase", letterSpacing: "0.07em" }}>
        {label}
      </span>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [dark, setDark]                     = useState(false);
  const [painPoints, setPainPoints]         = useState([]);
  const [allPainPoints, setAllPainPoints]   = useState([]);
  const [solutions, setSolutions]           = useState({});
  const [loading, setLoading]               = useState({});
  const [activeSolution, setActiveSolution] = useState(null);
  const [fetchingPosts, setFetchingPosts]   = useState(true);
  const [searchTerm, setSearchTerm]         = useState("");
  const [searchLoading, setSearchLoading]   = useState(false);
  const [selectedCat, setSelectedCat]       = useState("All");
  const [useMock, setUseMock]               = useState(false);
  const [stats, setStats]                   = useState({ total: 0, analyzed: 0, categories: 0 });
  const [searchFocused, setSearchFocused]   = useState(false);

  const t = T[dark ? "dark" : "light"];

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/pain-points?limit=100&min_score=5`);
        if (!res.ok) throw new Error("API offline");
        const data = await res.json();
        const pts = data.pain_points || [];
        const list = pts.length ? pts : MOCK_PAIN_POINTS;
        setPainPoints(list);
        setAllPainPoints(list);
        setUseMock(pts.length === 0);
        const cats = new Set(pts.map(p => p.category).filter(Boolean));
        setStats({ total: pts.length, analyzed: 0, categories: cats.size });
      } catch {
        setPainPoints(MOCK_PAIN_POINTS);
        setAllPainPoints(MOCK_PAIN_POINTS);
        setUseMock(true);
        setStats({ total: MOCK_PAIN_POINTS.length, analyzed: 0, categories: 5 });
      } finally {
        setFetchingPosts(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!searchTerm || searchTerm.length < 3) { setPainPoints(allPainPoints); return; }
    if (useMock) { setPainPoints(allPainPoints.filter(p => p.title.toLowerCase().includes(searchTerm.toLowerCase()))); return; }
    const timer = setTimeout(async () => {
      setSearchLoading(true);
      try {
        const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(searchTerm)}&limit=50`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        const results = data.results || [];
        setPainPoints(results.length > 0 ? results : allPainPoints.filter(p => p.title.toLowerCase().includes(searchTerm.toLowerCase())));
      } catch {
        setPainPoints(allPainPoints.filter(p => p.title.toLowerCase().includes(searchTerm.toLowerCase())));
      } finally { setSearchLoading(false); }
    }, 400);
    return () => clearTimeout(timer);
  }, [searchTerm, allPainPoints, useMock]);

  const analyze = useCallback(async (point) => {
    const key = point.title;
    if (solutions[key]) { setActiveSolution(key); return; }
    setLoading(l => ({ ...l, [key]: true }));
    setActiveSolution(key);
    try {
      if (useMock) {
        await new Promise(r => setTimeout(r, 1600));
        setSolutions(s => ({ ...s, [key]: { pain_point: key, category: point.category, analysis: MOCK_SOLUTION } }));
      } else {
        const res = await fetch(`${API_BASE}/solutions/generate`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: point.title, category: point.category || "General", content: point.content || "", score: point.score || 0, num_comments: point.num_comments || 0 }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setSolutions(s => ({ ...s, [key]: data }));
      }
      setActiveSolution(key);
      setStats(s => ({ ...s, analyzed: s.analyzed + 1 }));
    } catch (err) { alert("Error: " + err.message); }
    finally { setLoading(l => ({ ...l, [key]: false })); }
  }, [solutions, useMock]);

  const categories = ["All", ...new Set(allPainPoints.map(p => p.category).filter(Boolean))];
  const filtered = painPoints.filter(p => selectedCat === "All" || p.category === selectedCat);

  const resultLabel = fetchingPosts
    ? "Loading…"
    : searchTerm.length >= 3
      ? `${filtered.length} result${filtered.length !== 1 ? "s" : ""} for "${searchTerm}"`
      : `${filtered.length} problem${filtered.length !== 1 ? "s" : ""}`;

  return (
    <div style={{ minHeight: "100vh", background: t.page, color: t.text, fontFamily: "'DM Sans', system-ui, sans-serif", transition: "background 0.25s, color 0.25s" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { from { opacity: 0.4 } to { opacity: 0.7 } }
        * { box-sizing: border-box; }
        button { cursor: pointer; font-family: inherit; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.15); border-radius: 2px; }
        input { outline: none; font-family: inherit; }
        input::placeholder { color: var(--ph); }
      `}</style>

      {/* ── HEADER ── */}
      <header style={{
        position: "sticky", top: 0, zIndex: 50,
        borderBottom: `1px solid ${t.border}`,
        background: t.header,
        backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
      }}>
        <div style={{ maxWidth: 1260, margin: "0 auto", padding: "0 24px", height: 52, display: "flex", alignItems: "center", gap: 0 }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginRight: 20 }}>
            <div style={{
              width: 26, height: 26, borderRadius: 7,
              background: t.text,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Zap size={12} color={t.page} fill={t.page} />
            </div>
            <span style={{
              fontSize: 13.5, fontWeight: 500, letterSpacing: "-0.02em",
              fontFamily: "'DM Sans', sans-serif", color: t.text,
            }}>
              OpportunityLens
            </span>
          </div>

          <span style={{ fontSize: 12, color: t.textTer, marginRight: "auto" }}>
            Demand Explorer
          </span>

          {useMock && (
            <div style={{
              display: "flex", alignItems: "center", gap: 5, padding: "3px 8px",
              borderRadius: 5, background: t.inner, border: `1px solid ${t.border}`,
              fontSize: 10.5, fontWeight: 500, color: t.textSec, marginRight: 16,
              fontFamily: "'DM Mono', monospace",
            }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#F59E0B" }} />
              sample data
            </div>
          )}

          {/* Stats */}
          <div style={{ display: "flex", gap: 0 }}>
            <StatBox label="tracked" value={stats.total} t={t} />
            <StatBox label="built" value={Object.keys(solutions).length} t={t} />
            <StatBox label="markets" value={categories.length - 1} t={t} />
          </div>

          {/* Theme toggle */}
          <button
            onClick={() => setDark(d => !d)}
            style={{
              marginLeft: 16, padding: "6px", borderRadius: 7,
              background: t.inner, border: `1px solid ${t.border}`,
              color: t.textSec, display: "flex", alignItems: "center",
              transition: "all 0.15s",
            }}
          >
            {dark ? <Sun size={14} /> : <Moon size={14} />}
          </button>
        </div>
      </header>

      {/* ── MAIN ── */}
      <main style={{ maxWidth: 1260, margin: "0 auto", padding: "40px 24px 60px" }}>

        {/* Hero */}
        <div style={{ marginBottom: 36 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            padding: "3px 10px", borderRadius: 5,
            background: t.inner, border: `1px solid ${t.border}`,
            fontSize: 11, color: t.textSec, marginBottom: 14,
            fontFamily: "'DM Mono', monospace",
          }}>
            <span style={{ width: 4, height: 4, borderRadius: "50%", background: "#10B981" }} />
            built for founders who start from real demand
          </div>

          <h1 style={{
            fontSize: "clamp(24px, 3vw, 34px)", fontWeight: 400,
            margin: "0 0 12px", letterSpacing: "-0.03em",
            fontFamily: "'DM Serif Display', serif", lineHeight: 1.15,
            color: t.text,
          }}>
            Reddit pain points →{" "}
            <em style={{ fontStyle: "italic", color: t.textSec }}>real businesses.</em>
          </h1>

          <p style={{ fontSize: 13.5, color: t.textSec, margin: 0, maxWidth: 420, lineHeight: 1.75, fontWeight: 400 }}>
            Stop guessing what to build. These are problems people actively vent about — with real frustration, real volume, and real money waiting.
          </p>
        </div>

        {/* Search + filters */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ position: "relative", maxWidth: 380, marginBottom: 10 }}>
            <Search size={12} style={{
              position: "absolute", left: 11, top: "50%", transform: "translateY(-50%)",
              color: searchFocused ? t.text : t.textTer, pointerEvents: "none", transition: "color 0.15s",
            }} />
            <input
              placeholder="Search problems…"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              onFocus={() => setSearchFocused(true)}
              onBlur={() => setSearchFocused(false)}
              style={{
                width: "100%", paddingLeft: 30, paddingRight: 14, paddingTop: 8, paddingBottom: 8,
                borderRadius: 8,
                background: t.inputBg,
                border: `1px solid ${searchFocused ? t.borderStrong : t.border}`,
                boxShadow: searchFocused ? `0 0 0 2px ${t.border}` : "none",
                color: t.text, fontSize: 13,
                transition: "all 0.15s",
              }}
            />
            {searchLoading && (
              <div style={{
                position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)",
                width: 12, height: 12, borderRadius: "50%",
                border: `1.5px solid ${t.border}`, borderTopColor: t.text,
                animation: "spin 0.9s linear infinite",
              }} />
            )}
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {categories.map(cat => (
              <FilterPill
                key={cat} cat={cat} c={getCat(cat, dark)}
                active={selectedCat === cat} isAll={cat === "All"}
                onClick={() => setSelectedCat(cat)} dark={dark}
              />
            ))}
          </div>
        </div>

        {/* Two-column layout */}
        <div style={{ display: "grid", gridTemplateColumns: "minmax(300px, 380px) 1fr", gap: "0 20px", alignItems: "start" }}>

          {/* Left — problem list */}
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
              <span style={{ fontSize: 11, color: t.textTer, fontFamily: "'DM Mono', monospace" }}>
                {resultLabel}
              </span>
              {!fetchingPosts && filtered.length > 0 && !searchTerm && (
                <span style={{ fontSize: 10.5, color: t.textTer, display: "flex", alignItems: "center", gap: 3 }}>
                  <Clock size={8} />sorted by demand
                </span>
              )}
            </div>

            {fetchingPosts ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                {[0, 1, 2].map(i => <SkeletonCard key={i} t={t} />)}
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                <AnimatePresence>
                  {filtered.map(point => (
                    <PainCard
                      key={point.title} point={point} onAnalyze={analyze}
                      isLoading={!!loading[point.title]}
                      isAnalyzed={!!solutions[point.title]}
                      isActive={activeSolution === point.title}
                      dark={dark}
                    />
                  ))}
                </AnimatePresence>

                {filtered.length === 0 && (
                  <div style={{
                    padding: "44px 20px", display: "flex", flexDirection: "column",
                    alignItems: "center", gap: 8,
                    border: `1px dashed ${t.border}`, borderRadius: 10,
                  }}>
                    <Search size={16} style={{ color: t.textTer }} />
                    <p style={{ fontSize: 13, color: t.textSec, margin: 0 }}>
                      {searchTerm.length >= 3
                        ? `No results for "${searchTerm}"`
                        : "No problems match this filter"}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right — solution panel */}
          <div style={{ position: "sticky", top: 66, display: "flex", flexDirection: "column", gap: 10 }}>
            <AnimatePresence mode="wait">
              {!activeSolution ? (
                <EmptyState key="empty" t={t} />
              ) : (loading[activeSolution] && !solutions[activeSolution]) ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  style={{ borderRadius: 12, border: `1px solid ${t.border}`, background: t.panel, overflow: "hidden" }}
                >
                  <div style={{ height: 3, background: t.borderMid, animation: "pulse 1.2s ease-in-out infinite alternate" }} />
                  <Spinner t={t} />
                </motion.div>
              ) : solutions[activeSolution] ? (
                <SolutionPanel
                  key={activeSolution}
                  data={solutions[activeSolution]}
                  painPoint={allPainPoints.find(p => p.title === activeSolution)}
                  dark={dark}
                />
              ) : null}
            </AnimatePresence>

            {/* History */}
            {Object.keys(solutions).length > 0 && (
              <div style={{ paddingTop: 2 }}>
                <p style={{ fontSize: 10, fontWeight: 500, color: t.textTer, marginBottom: 5, paddingLeft: 2, fontFamily: "'DM Mono', monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Recent
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {Object.keys(solutions).map(key => (
                    <HistoryItem
                      key={key} histKey={key}
                      isActive={activeSolution === key}
                      allPainPoints={allPainPoints}
                      onClick={() => setActiveSolution(key)}
                      dark={dark}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer style={{ borderTop: `1px solid ${t.border}`, padding: "16px 24px" }}>
        <div style={{ maxWidth: 1260, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
          <span style={{ fontSize: 11.5, color: t.textTer, fontFamily: "'DM Mono', monospace" }}>
            OpportunityLens — where real problems become real businesses
          </span>
          <div style={{ display: "flex", gap: 14, fontSize: 11, color: t.textTer }}>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}><Shield size={9} />Sourced from Reddit</span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}><Layers size={9} />Ranked by signal</span>
          </div>
        </div>
      </footer>
    </div>
  );
}