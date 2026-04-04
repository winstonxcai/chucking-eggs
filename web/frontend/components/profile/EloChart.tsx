"use client";

import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface EloPoint {
  date: string;
  elo: number;
}

interface EloChartProps {
  data: EloPoint[];
}

export default function EloChart({ data }: EloChartProps) {
  if (data.length === 0) {
    return (
      <div className="py-2 text-center text-sm text-text-secondary">
        No games yet
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="eloFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#D97757" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#D97757" stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: "#8C8478" }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={["auto", "auto"]}
          tick={{ fontSize: 11, fill: "#8C8478" }}
          tickLine={false}
          axisLine={false}
          width={36}
        />
        <Tooltip
          contentStyle={{
            background: "#FFFFFF",
            border: "1px solid #E8E2D9",
            borderRadius: "8px",
            fontSize: "12px",
            color: "#1A1612",
          }}
          formatter={(v) => [v as number, "Elo"]}
        />
        <Area
          type="monotone"
          dataKey="elo"
          stroke="#D97757"
          strokeWidth={2}
          fill="url(#eloFill)"
          dot={false}
          activeDot={{ r: 4, fill: "#D97757" }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
