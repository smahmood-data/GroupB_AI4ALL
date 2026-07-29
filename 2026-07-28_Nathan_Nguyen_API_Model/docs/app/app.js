"use strict";

/**
 * Bilingual, dependency-free dashboard behavior.
 *
 * The page uses a generated public snapshot rather than calling health APIs
 * from every visitor's browser. That keeps the displayed forecast auditable
 * and lets the same app work from GitHub Pages or directly from a local file.
 */

const DATA_URL = "./data/dashboard.json";
const LANGUAGE_KEY = "dengue-signal-language";

const translations = {
  en: {
    skip: "Skip to the dashboard",
    languageLabel: "Language",
    modelExplainer: "How the model works",
    researchNoticeLabel: "Important notice",
    researchTag: "Research estimate",
    researchNotice: "Not an official outbreak declaration or medical guidance.",
    officialGuidance: "Official guidance ↗",
    loading: "Loading the latest Puerto Rico update…",
    errorTitle: "The latest update is unavailable.",
    errorBody: "We could not load the public forecast. Please try again later.",
    seeFreshness: "See the source dates",
    islandWide: "Puerto Rico · island-wide",
    outbreakChance: "Chance of an outbreak",
    forecastWeekLabel: "Forecast week: {range}",
    probabilityExplanation: "{probability} is the model’s estimated chance that reported cases will exceed {cutoff} during {range}. In this app, that counts as an outbreak because {cutoff} is the seasonal top-25% cutoff—not an official declaration.",
    latestDetailsLabel: "Latest weekly estimates",
    estimatedCases: "Starting estimate",
    cases: "Cases",
    highEstimateRange: "Higher estimate range",
    outbreakCutoff: "Outbreak cutoff",
    mainEstimateDefinition: "Starting estimate means the latest official weekly count, from {sourceRange}, carried forward. This simple fallback was more accurate in testing than adjusting the count. Its average error was {error} cases per week. It is not a confirmed count for {forecastRange}.",
    highRangeDefinition: "The app does not calculate a reliable chance of landing inside this range. Across {weeks} test weeks, actual cases were at or below the lower estimate in {lower} of weeks and the upper estimate in {upper}.",
    unusualLevelDefinition: "More than {cutoff} cases would count as an outbreak in this app because that is above the seasonal top-25% cutoff.",
    dataCurrentness: "How current are the data?",
    checkDates: "Check all dates →",
    trendKicker: "Explore the history",
    trendTitle: "Move through Puerto Rico’s weekly dengue data",
    trendDescription: "Drag across the chart or tap a week to inspect exact values. The visible gap before the estimate reflects delayed official reporting.",
    showLabel: "Show",
    metricLabel: "Chart measure",
    hospitalizations: "Hospitalizations",
    timeRangeLabel: "Time range",
    threeMonths: "3 months",
    oneYear: "1 year",
    allHistory: "All history",
    chartLegend: "Chart legend",
    reportedCases: "Reported cases",
    mostLikelyEstimate: "Starting estimate",
    higherPossibleRange: "Higher estimate range",
    chartAriaCases: "Interactive line chart of reported dengue cases and the latest estimates",
    chartAriaHospitalized: "Interactive line chart of reported dengue hospitalizations",
    selectedWeek: "Selected week",
    chartInstruction: "Drag across the chart, tap a week, or use the left and right arrow keys to see exact values.",
    showTable: "Show the chart as a table",
    week: "Week",
    reportedValue: "Reported cases",
    hospitalizedValue: "Hospitalizations",
    confirmedPcr: "Cases confirmed by PCR",
    higherPossible: "Higher estimate range",
    officialWeekNote: "These are official values reported for this week.",
    estimateWeekNote: "These are estimates, not confirmed counts. Official reports have not reached this week yet.",
    noHospitalEstimate: "Hospitalization estimates are not available for future weeks.",
    chartSummaryCases: "For {range}, the starting estimate is {mostLikely} cases and the higher estimate range is {higher} to {highest}. The latest official week reported {actual} cases.",
    chartSummaryOneHospitalized: "The latest official week reported 1 dengue hospitalization.",
    chartSummaryManyHospitalized: "The latest official week reported {count} dengue hospitalizations.",
    freshnessKicker: "Check the source dates",
    freshnessTitle: "What information was available for this update",
    freshnessDescription: "The app always shows when official reports stop. Older inputs are never presented as current.",
    latestCaseWeek: "Latest official case week",
    ageWeeks: "{count} weeks before this estimate",
    weatherCoverage: "Weather coverage",
    days: "{count} days",
    weatherCaption: "days available for this update",
    forecastIssued: "Estimate created",
    sourceAvailable: "official source was reachable",
    sourceUnavailable: "official source was unavailable",
    trainingCutoff: "Data used to train this version",
    trainingCaption: "new reports are added only after safety checks",
    staleTitle: "Official case reports are delayed",
    staleBody: "The latest complete case week is {date}, about {count} weeks before this estimate.",
    delayedTitle: "Official case reports have a short delay",
    delayedBody: "The latest complete case week is {date}, about {count} weeks before this estimate.",
    currentTitle: "Official case data are current",
    currentBody: "The latest complete case week is {date}.",
    unavailableTitle: "The latest official case date is unavailable",
    unavailableBody: "Use caution until the source status is restored.",
    reportsBehind: "Reports are {count} weeks behind",
    reportsShortDelay: "Reports are {count} weeks behind",
    reportsCurrent: "Reports are current",
    reportsUnavailable: "Report date unavailable",
    dataAgeExplanation: "The estimate uses the newest official information available and clearly shows the reporting gap.",
    methodKicker: "Want the technical details?",
    methodTitle: "See the data sources, model design, and testing results.",
    openExplainer: "Open the model explainer →",
    footerDescription: "An experimental forecasting project using public health and weather data.",
    dataSource: "Puerto Rico health data ↗",
    methodology: "How the model works",
    closeLabel: "Close",
    dialogKicker: "How the estimate is made",
    dialogTitle: "Three kinds of information shape the weekly estimate.",
    dialogCasesTitle: "Reported cases",
    dialogCasesBody: "Official Puerto Rico reports show how dengue activity changed in earlier weeks.",
    dialogWeatherTitle: "Weather and season",
    dialogWeatherBody: "Rain, temperature, humidity, soil moisture, and the time of year help identify conditions linked with dengue.",
    dialogRangeTitle: "A range, not one certainty",
    dialogRangeBody: "The app shows a starting estimate and a higher range because the future cannot be known exactly.",
    increasedTitle: "Dengue activity may increase next week",
    steadyTitle: "Dengue activity is not expected to increase next week",
    increasedSummary: "The latest data suggest a higher-than-usual week is possible. Follow official Puerto Rico guidance and use the chart below to understand the estimate.",
    steadySummary: "The latest data do not suggest an increase, but dengue is still present. Continue following official Puerto Rico guidance."
  },
  es: {
    skip: "Saltar al tablero",
    languageLabel: "Idioma",
    modelExplainer: "Cómo funciona el modelo",
    researchNoticeLabel: "Aviso importante",
    researchTag: "Estimación de investigación",
    researchNotice: "No es una declaración oficial de brote ni orientación médica.",
    officialGuidance: "Orientación oficial ↗",
    loading: "Cargando la actualización más reciente de Puerto Rico…",
    errorTitle: "La actualización más reciente no está disponible.",
    errorBody: "No pudimos cargar el pronóstico público. Inténtelo de nuevo más tarde.",
    seeFreshness: "Ver las fechas de las fuentes",
    islandWide: "Puerto Rico · toda la isla",
    outbreakChance: "Probabilidad de un brote",
    forecastWeekLabel: "Semana estimada: {range}",
    probabilityExplanation: "{probability} es la probabilidad estimada por el modelo de que los casos informados superen {cutoff} durante {range}. En esta aplicación, eso cuenta como brote porque {cutoff} es el límite estacional del 25% más alto, no una declaración oficial.",
    latestDetailsLabel: "Estimaciones semanales más recientes",
    estimatedCases: "Estimación inicial",
    cases: "Casos",
    highEstimateRange: "Rango de estimaciones altas",
    outbreakCutoff: "Límite de brote",
    mainEstimateDefinition: "Estimación inicial significa que el conteo semanal oficial más reciente, de {sourceRange}, se mantiene como punto de partida. Esta alternativa sencilla fue más precisa en las pruebas que ajustar el conteo. Su error promedio fue de {error} casos por semana. No es un conteo confirmado para {forecastRange}.",
    highRangeDefinition: "La aplicación no calcula una probabilidad confiable de quedar dentro de este rango. En {weeks} semanas de prueba, los casos reales estuvieron en o por debajo de la estimación inferior en el {lower} de las semanas y de la superior en el {upper}.",
    unusualLevelDefinition: "Más de {cutoff} casos contaría como un brote en esta aplicación porque supera el límite estacional del 25% más alto.",
    dataCurrentness: "¿Qué tan recientes son los datos?",
    checkDates: "Ver todas las fechas →",
    trendKicker: "Explore el historial",
    trendTitle: "Recorra los datos semanales de dengue de Puerto Rico",
    trendDescription: "Arrastre por la gráfica o toque una semana para ver valores exactos. La brecha antes de la estimación refleja el atraso de los informes oficiales.",
    showLabel: "Mostrar",
    metricLabel: "Medida de la gráfica",
    hospitalizations: "Hospitalizaciones",
    timeRangeLabel: "Periodo",
    threeMonths: "3 meses",
    oneYear: "1 año",
    allHistory: "Todo el historial",
    chartLegend: "Leyenda de la gráfica",
    reportedCases: "Casos informados",
    mostLikelyEstimate: "Estimación inicial",
    higherPossibleRange: "Rango de estimaciones altas",
    chartAriaCases: "Gráfica interactiva de casos de dengue informados y las estimaciones más recientes",
    chartAriaHospitalized: "Gráfica interactiva de hospitalizaciones por dengue informadas",
    selectedWeek: "Semana seleccionada",
    chartInstruction: "Arrastre por la gráfica, toque una semana o use las flechas izquierda y derecha para ver valores exactos.",
    showTable: "Mostrar la gráfica como tabla",
    week: "Semana",
    reportedValue: "Casos informados",
    hospitalizedValue: "Hospitalizaciones",
    confirmedPcr: "Casos confirmados por PCR",
    higherPossible: "Rango de estimaciones altas",
    officialWeekNote: "Estos son valores oficiales informados para esta semana.",
    estimateWeekNote: "Estas son estimaciones, no conteos confirmados. Los informes oficiales todavía no han llegado a esta semana.",
    noHospitalEstimate: "No hay estimaciones de hospitalización para semanas futuras.",
    chartSummaryCases: "Para {range}, la estimación inicial es {mostLikely} casos y el rango de estimaciones altas es de {higher} a {highest}. La última semana oficial informó {actual} casos.",
    chartSummaryOneHospitalized: "La última semana oficial informó 1 hospitalización por dengue.",
    chartSummaryManyHospitalized: "La última semana oficial informó {count} hospitalizaciones por dengue.",
    freshnessKicker: "Revise las fechas de las fuentes",
    freshnessTitle: "La información disponible para esta actualización",
    freshnessDescription: "La aplicación siempre muestra dónde terminan los informes oficiales. Nunca presenta información antigua como si fuera actual.",
    latestCaseWeek: "Última semana oficial de casos",
    ageWeeks: "{count} semanas antes de esta estimación",
    weatherCoverage: "Cobertura climática",
    days: "{count} días",
    weatherCaption: "días disponibles para esta actualización",
    forecastIssued: "Estimación creada",
    sourceAvailable: "la fuente oficial estuvo disponible",
    sourceUnavailable: "la fuente oficial no estuvo disponible",
    trainingCutoff: "Datos usados para entrenar esta versión",
    trainingCaption: "los informes nuevos se añaden solo después de controles de seguridad",
    staleTitle: "Los informes oficiales de casos están atrasados",
    staleBody: "La última semana completa de casos es {date}, aproximadamente {count} semanas antes de esta estimación.",
    delayedTitle: "Los informes oficiales tienen un breve atraso",
    delayedBody: "La última semana completa de casos es {date}, aproximadamente {count} semanas antes de esta estimación.",
    currentTitle: "Los datos oficiales de casos están al día",
    currentBody: "La última semana completa de casos es {date}.",
    unavailableTitle: "La fecha oficial más reciente no está disponible",
    unavailableBody: "Use precaución hasta que se restablezca el estado de la fuente.",
    reportsBehind: "Los informes tienen {count} semanas de atraso",
    reportsShortDelay: "Los informes tienen {count} semanas de atraso",
    reportsCurrent: "Los informes están al día",
    reportsUnavailable: "Fecha del informe no disponible",
    dataAgeExplanation: "La estimación usa la información oficial más reciente disponible y muestra claramente la brecha de reporte.",
    methodKicker: "¿Quiere los detalles técnicos?",
    methodTitle: "Vea las fuentes, el diseño del modelo y los resultados de las pruebas.",
    openExplainer: "Abrir la explicación del modelo →",
    footerDescription: "Un proyecto experimental de pronóstico que usa datos públicos de salud y clima.",
    dataSource: "Datos de salud de Puerto Rico ↗",
    methodology: "Cómo funciona el modelo",
    closeLabel: "Cerrar",
    dialogKicker: "Cómo se hace la estimación",
    dialogTitle: "Tres tipos de información forman la estimación semanal.",
    dialogCasesTitle: "Casos informados",
    dialogCasesBody: "Los informes oficiales de Puerto Rico muestran cómo cambió la actividad del dengue en semanas anteriores.",
    dialogWeatherTitle: "Clima y temporada",
    dialogWeatherBody: "La lluvia, temperatura, humedad, humedad del suelo y época del año ayudan a identificar condiciones relacionadas con el dengue.",
    dialogRangeTitle: "Un rango, no una certeza",
    dialogRangeBody: "La aplicación muestra una estimación inicial y un rango alto porque el futuro no se puede conocer con exactitud.",
    increasedTitle: "La actividad del dengue podría aumentar la próxima semana",
    steadyTitle: "No se espera un aumento en la actividad del dengue la próxima semana",
    increasedSummary: "Los datos más recientes indican que una semana más alta de lo usual es posible. Siga la orientación oficial de Puerto Rico y use la gráfica para entender la estimación.",
    steadySummary: "Los datos más recientes no indican un aumento, pero el dengue sigue presente. Continúe siguiendo la orientación oficial de Puerto Rico."
  }
};

let dashboardData = null;
let language = localStorage.getItem(LANGUAGE_KEY) === "es" ? "es" : "en";
let chartMetric = "cases";
let chartRange = 52;
let selectedChartIndex = null;
let activeChartRows = [];
let chartGeometry = null;
let isDraggingChart = false;
let chartResizeObserver = null;

function t(key, replacements = {}) {
  let value = translations[language][key] || translations.en[key] || key;
  Object.entries(replacements).forEach(([name, replacement]) => {
    value = value.replaceAll(`{${name}}`, String(replacement));
  });
  return value;
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat(language === "es" ? "es-PR" : "en-US", {
    style: "percent",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  }).format(value);
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat(language === "es" ? "es-PR" : "en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  }).format(value);
}

function formatDate(value, options = { month: "short", day: "numeric", year: "numeric" }) {
  if (!value) return "—";
  const parsed = new Date(`${value.slice(0, 10)}T12:00:00`);
  return new Intl.DateTimeFormat(language === "es" ? "es-PR" : "en-US", options).format(parsed);
}

function formatWeekRange(value) {
  if (!value) return "—";
  const start = new Date(`${value.slice(0, 10)}T12:00:00`);
  const end = new Date(start);
  end.setDate(start.getDate() + 6);
  const formatter = new Intl.DateTimeFormat(
    language === "es" ? "es-PR" : "en-US",
    { month: "short", day: "numeric", year: "numeric" }
  );
  if (typeof formatter.formatRange === "function") {
    return formatter.formatRange(start, end);
  }
  return `${formatter.format(start)} – ${formatter.format(end)}`;
}

function applyStaticTranslations() {
  document.documentElement.lang = language;
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
  });
  document.querySelectorAll(".language-button").forEach((button) => {
    const selected = button.dataset.language === language;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function updateFreshness() {
  const freshness = dashboardData.freshness;
  const age = freshness.officialCaseAgeWeeks;
  const latestDate = formatDate(freshness.latestOfficialCaseWeek);
  const banner = document.getElementById("freshnessBanner");
  banner.classList.toggle("is-current", freshness.level === "current");

  let titleKey = "unavailableTitle";
  let bodyKey = "unavailableBody";
  let ageHeadingKey = "reportsUnavailable";
  if (freshness.level === "current") {
    titleKey = "currentTitle";
    bodyKey = "currentBody";
    ageHeadingKey = "reportsCurrent";
  } else if (freshness.level === "delayed") {
    titleKey = "delayedTitle";
    bodyKey = "delayedBody";
    ageHeadingKey = "reportsShortDelay";
  } else if (freshness.level === "stale") {
    titleKey = "staleTitle";
    bodyKey = "staleBody";
    ageHeadingKey = "reportsBehind";
  }

  document.getElementById("freshnessBannerTitle").textContent = t(titleKey);
  document.getElementById("freshnessBannerBody").textContent = t(bodyKey, {
    date: latestDate,
    count: formatNumber(age)
  });
  document.getElementById("dataAgeHeading").textContent = t(ageHeadingKey, {
    count: formatNumber(age)
  });
  document.getElementById("dataAgeExplanation").textContent = t("dataAgeExplanation");
  document.getElementById("latestCaseWeek").textContent = latestDate;
  document.getElementById("caseAgeText").textContent = t("ageWeeks", {
    count: formatNumber(age)
  });
  document.getElementById("weatherDays").textContent = t("days", {
    count: formatNumber(freshness.weatherDaysAvailable)
  });
  document.getElementById("issuedDate").textContent = formatDate(dashboardData.issuedAt);
  document.getElementById("sourceStatusText").textContent =
    freshness.sourceStatus === "ok" ? t("sourceAvailable") : t("sourceUnavailable");
  document.getElementById("trainingCutoff").textContent =
    formatDate(dashboardData.model.trainingDataCutoff);
}

function updateRangeVisual(cases) {
  const maximum = Math.max(cases.p90 || 0, cases.seasonalThreshold || 0, 1) * 1.12;
  const position = (value) =>
    Math.min(100, Math.max(0, (value / maximum) * 100));
  const highStart = position(cases.p80 || 0);
  const highEnd = position(cases.p90 || 0);
  document.getElementById("p50Dot").style.left =
    `${position(cases.p50 || 0)}%`;
  document.getElementById("highRangeBand").style.left = `${highStart}%`;
  document.getElementById("highRangeBand").style.width =
    `${Math.max(0, highEnd - highStart)}%`;
  document.getElementById("thresholdMarker").style.left =
    `${position(cases.seasonalThreshold || 0)}%`;
}

function updateDynamicContent() {
  if (!dashboardData) return;
  const forecast = dashboardData.signal.nextWeek;
  const cases = forecast.cases;
  const increased = forecast.outbreak.alert === true;
  const forecastRange = formatWeekRange(forecast.weekStart);

  document.getElementById("forecastWeek").textContent =
    t("forecastWeekLabel", { range: forecastRange });
  document.getElementById("probabilityWeek").textContent = forecastRange;
  document.getElementById("caseForecastWeek").textContent =
    t("forecastWeekLabel", { range: forecastRange });
  document.getElementById("signalHeading").textContent =
    increased ? t("increasedTitle") : t("steadyTitle");
  document.getElementById("signalSummary").textContent =
    increased ? t("increasedSummary") : t("steadySummary");
  document.getElementById("probabilityValue").textContent =
    formatPercent(forecast.outbreak.probability);
  document.getElementById("probabilityExplanation").textContent =
    t("probabilityExplanation", {
      probability: formatPercent(forecast.outbreak.probability),
      cutoff: formatNumber(cases.seasonalThreshold),
      range: forecastRange
    });
  document.getElementById("signalDisc").classList.toggle("is-alert", increased);

  document.getElementById("p50Value").textContent = formatNumber(cases.p50);
  document.getElementById("mainEstimateDefinition").textContent =
    t("mainEstimateDefinition", {
      sourceRange: formatWeekRange(
        dashboardData.freshness.latestOfficialCaseWeek
      ),
      error: formatNumber(dashboardData.model.heldOut.caseMae),
      forecastRange
    });
  document.getElementById("highRangeValue").textContent =
    `${formatNumber(cases.p80)}–${formatNumber(cases.p90)} ${t("cases").toLowerCase()}`;
  document.getElementById("unusualLevelValue").textContent =
    `${formatNumber(cases.seasonalThreshold)}+ ${t("cases").toLowerCase()}`;
  document.getElementById("unusualLevelDefinition").textContent =
    t("unusualLevelDefinition", {
      cutoff: formatNumber(cases.seasonalThreshold)
    });
  document.getElementById("highRangeDefinition").textContent = t("highRangeDefinition", {
    weeks: formatNumber(dashboardData.model.heldOut.caseTestWeeks),
    lower: formatPercent(
      dashboardData.model.heldOut.lowerHighEstimateCoverage, 0
    ),
    upper: formatPercent(
      dashboardData.model.heldOut.upperHighEstimateCoverage, 0
    )
  });
  updateRangeVisual(cases);
  updateFreshness();
  renderChart();
}

function allChartRows() {
  const reported = dashboardData.history.map((row) => ({
    type: "reported",
    date: row.weekStart,
    cases: row.totalCases,
    hospitalized: row.hospitalizedCases,
    pcr: row.pcrCases,
    mostLikely: null,
    higher: null,
    highest: null
  }));
  const estimated = [
    dashboardData.signal.currentWeek,
    dashboardData.signal.nextWeek
  ].map((row) => ({
    type: "estimate",
    date: row.weekStart,
    cases: null,
    hospitalized: null,
    pcr: null,
    mostLikely: row.cases.p50,
    higher: row.cases.p80,
    highest: row.cases.p90
  }));
  return { reported, estimated };
}

function filteredChartRows() {
  const { reported, estimated } = allChartRows();
  const visibleReported = chartRange === "all" ? reported : reported.slice(-chartRange);
  return chartMetric === "cases"
    ? visibleReported.concat(estimated)
    : visibleReported;
}

function chartValue(row) {
  if (row.type === "estimate") return row.mostLikely;
  return chartMetric === "cases" ? row.cases : row.hospitalized;
}

function drawChart() {
  const canvas = document.getElementById("trendChart");
  const container = canvas.parentElement;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(container.clientWidth, 320);
  const height = Math.max(container.clientHeight, 260);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);

  const padding = { top: 24, right: 20, bottom: 44, left: 48 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = activeChartRows.flatMap((row) => (
    chartMetric === "cases"
      ? [row.cases, row.mostLikely, row.highest]
      : [row.hospitalized]
  )).filter((value) => value !== null && value !== undefined);
  const rawMax = Math.max(...values, 10);
  const tickSize = rawMax > 100 ? 50 : rawMax > 40 ? 20 : 10;
  const axisMax = Math.ceil(rawMax / tickSize) * tickSize;
  const denominator = Math.max(activeChartRows.length - 1, 1);
  const x = (index) => padding.left + (index / denominator) * plotWidth;
  const y = (value) => padding.top + plotHeight - ((value || 0) / axisMax) * plotHeight;
  const css = getComputedStyle(document.documentElement);

  chartGeometry = { padding, plotWidth, width, x };
  context.clearRect(0, 0, width, height);
  context.font = "11px Inter, sans-serif";
  context.lineWidth = 1;
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = (axisMax / 4) * tick;
    const tickY = y(value);
    context.strokeStyle = "#e5e1d8";
    context.beginPath();
    context.moveTo(padding.left, tickY);
    context.lineTo(width - padding.right, tickY);
    context.stroke();
    context.fillStyle = css.getPropertyValue("--muted");
    context.fillText(formatNumber(value), padding.left - 9, tickY);
  }

  const reportedCount = activeChartRows.filter((row) => row.type === "reported").length;
  if (chartMetric === "cases" && reportedCount && reportedCount < activeChartRows.length) {
    const gapStart = x(reportedCount - 1);
    const gapEnd = x(reportedCount);
    context.fillStyle = "rgba(196, 145, 46, .09)";
    context.fillRect(gapStart, padding.top, gapEnd - gapStart, plotHeight);
    context.setLineDash([4, 5]);
    context.strokeStyle = "rgba(196, 145, 46, .65)";
    context.beginPath();
    context.moveTo(gapEnd, padding.top);
    context.lineTo(gapEnd, padding.top + plotHeight);
    context.stroke();
    context.setLineDash([]);
  }

  context.strokeStyle = css.getPropertyValue("--ink");
  context.lineWidth = 2.5;
  context.beginPath();
  activeChartRows.forEach((row, index) => {
    if (row.type !== "reported") return;
    const value = chartMetric === "cases" ? row.cases : row.hospitalized;
    if (index === 0) context.moveTo(x(index), y(value));
    else context.lineTo(x(index), y(value));
  });
  context.stroke();

  if (chartMetric === "cases") {
    const estimateRows = activeChartRows.filter((row) => row.type === "estimate");
    const estimateStart = activeChartRows.length - estimateRows.length;
    context.fillStyle = "rgba(217, 103, 79, .18)";
    context.beginPath();
    estimateRows.forEach((row, offset) => {
      const pointX = x(estimateStart + offset);
      const pointY = y(row.highest);
      if (offset === 0) context.moveTo(pointX, pointY);
      else context.lineTo(pointX, pointY);
    });
    estimateRows.slice().reverse().forEach((row, reverseOffset) => {
      const offset = estimateRows.length - 1 - reverseOffset;
      context.lineTo(x(estimateStart + offset), y(row.higher));
    });
    context.closePath();
    context.fill();

    context.strokeStyle = css.getPropertyValue("--coral");
    context.lineWidth = 3;
    context.beginPath();
    estimateRows.forEach((row, offset) => {
      const pointX = x(estimateStart + offset);
      const pointY = y(row.mostLikely);
      if (offset === 0) context.moveTo(pointX, pointY);
      else context.lineTo(pointX, pointY);
    });
    context.stroke();
  }

  const safeSelected = Math.min(
    Math.max(selectedChartIndex ?? activeChartRows.length - 1, 0),
    activeChartRows.length - 1
  );
  selectedChartIndex = safeSelected;
  const selectedRow = activeChartRows[safeSelected];
  const selectedX = x(safeSelected);
  const selectedValue = chartValue(selectedRow);
  context.strokeStyle = css.getPropertyValue("--teal");
  context.lineWidth = 1.5;
  context.setLineDash([3, 4]);
  context.beginPath();
  context.moveTo(selectedX, padding.top);
  context.lineTo(selectedX, padding.top + plotHeight);
  context.stroke();
  context.setLineDash([]);
  context.beginPath();
  context.fillStyle = selectedRow.type === "estimate"
    ? css.getPropertyValue("--coral")
    : css.getPropertyValue("--teal");
  context.arc(selectedX, y(selectedValue), 5, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = css.getPropertyValue("--muted");
  context.textAlign = "center";
  context.textBaseline = "top";
  const labelIndexes = [
    0,
    Math.floor((activeChartRows.length - 1) / 2),
    Math.max(reportedCount - 1, 0),
    activeChartRows.length - 1
  ];
  [...new Set(labelIndexes)].forEach((index) => {
    context.fillText(
      formatDate(activeChartRows[index].date, { month: "short", year: "2-digit" }),
      x(index),
      height - padding.bottom + 13
    );
  });

  updateSelectedWeekCard(selectedRow);
}

function updateSelectedWeekCard(row) {
  document.getElementById("selectedWeekDate").textContent = formatWeekRange(row.date);
  const secondaryBlock = document.getElementById("selectedSecondaryBlock");
  const tertiaryBlock = document.getElementById("selectedTertiaryBlock");
  secondaryBlock.hidden = false;
  tertiaryBlock.hidden = false;

  if (row.type === "estimate") {
    document.getElementById("selectedPrimaryLabel").textContent = t("mostLikelyEstimate");
    document.getElementById("selectedPrimaryValue").textContent =
      `${formatNumber(row.mostLikely)} ${t("cases").toLowerCase()}`;
    document.getElementById("selectedSecondaryLabel").textContent = t("higherPossible");
    document.getElementById("selectedSecondaryValue").textContent =
      `${formatNumber(row.higher)}–${formatNumber(row.highest)} ${t("cases").toLowerCase()}`;
    tertiaryBlock.hidden = true;
    document.getElementById("selectedWeekNote").textContent = t("estimateWeekNote");
    return;
  }

  if (chartMetric === "hospitalized") {
    document.getElementById("selectedPrimaryLabel").textContent = t("hospitalizedValue");
    document.getElementById("selectedPrimaryValue").textContent =
      formatNumber(row.hospitalized);
    document.getElementById("selectedSecondaryLabel").textContent = t("reportedValue");
    document.getElementById("selectedSecondaryValue").textContent =
      formatNumber(row.cases);
    tertiaryBlock.hidden = true;
  } else {
    document.getElementById("selectedPrimaryLabel").textContent = t("reportedValue");
    document.getElementById("selectedPrimaryValue").textContent =
      formatNumber(row.cases);
    document.getElementById("selectedSecondaryLabel").textContent = t("hospitalizedValue");
    document.getElementById("selectedSecondaryValue").textContent =
      formatNumber(row.hospitalized);
    document.getElementById("selectedTertiaryLabel").textContent = t("confirmedPcr");
    document.getElementById("selectedTertiaryValue").textContent =
      formatNumber(row.pcr);
  }
  document.getElementById("selectedWeekNote").textContent = t("officialWeekNote");
}

function renderChartTable() {
  const body = document.getElementById("chartTableBody");
  body.textContent = "";
  activeChartRows.forEach((row) => {
    const tr = document.createElement("tr");
    const values = [
      formatWeekRange(row.date),
      row.cases,
      row.hospitalized,
      row.mostLikely,
      row.type === "estimate"
        ? `${formatNumber(row.higher)}–${formatNumber(row.highest)}`
        : null
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent =
        value === null || value === undefined
          ? "—"
          : (typeof value === "number" ? formatNumber(value) : value);
      tr.appendChild(cell);
    });
    body.appendChild(tr);
  });
}

function updateChartSummary() {
  const latestOfficial = dashboardData.history[dashboardData.history.length - 1];
  const nextWeek = dashboardData.signal.nextWeek;
  const summary = chartMetric === "cases"
    ? t("chartSummaryCases", {
        range: formatWeekRange(nextWeek.weekStart),
        actual: formatNumber(latestOfficial.totalCases),
        mostLikely: formatNumber(nextWeek.cases.p50),
        higher: formatNumber(nextWeek.cases.p80),
        highest: formatNumber(nextWeek.cases.p90)
      })
    : latestOfficial.hospitalizedCases === 1
      ? t("chartSummaryOneHospitalized")
      : t("chartSummaryManyHospitalized", {
          count: formatNumber(latestOfficial.hospitalizedCases)
        });
  document.getElementById("chartSummary").textContent = summary;
}

function renderChart() {
  if (!dashboardData) return;
  activeChartRows = filteredChartRows();
  if (
    selectedChartIndex === null
    || selectedChartIndex >= activeChartRows.length
  ) {
    selectedChartIndex = activeChartRows.length - 1;
  }

  const first = activeChartRows[0];
  const last = activeChartRows[activeChartRows.length - 1];
  document.getElementById("chartDateRange").textContent =
    `${formatDate(first.date, { month: "short", year: "numeric" })} — ${formatDate(last.date, { month: "short", year: "numeric" })}`;
  document.getElementById("trendChart").setAttribute(
    "aria-label",
    t(chartMetric === "cases" ? "chartAriaCases" : "chartAriaHospitalized")
  );
  document.getElementById("forecastLegend").hidden = chartMetric !== "cases";
  document.getElementById("rangeLegend").hidden = chartMetric !== "cases";
  renderChartTable();
  updateChartSummary();
  drawChart();

  if (!chartResizeObserver) {
    chartResizeObserver = new ResizeObserver(drawChart);
    chartResizeObserver.observe(document.querySelector(".chart-wrap"));
  }
}

function selectChartPoint(clientX) {
  if (!chartGeometry || !activeChartRows.length) return;
  const canvas = document.getElementById("trendChart");
  const rect = canvas.getBoundingClientRect();
  const localX = clientX - rect.left;
  const plotPosition = (localX - chartGeometry.padding.left) / chartGeometry.plotWidth;
  selectedChartIndex = Math.round(
    Math.min(1, Math.max(0, plotPosition)) * (activeChartRows.length - 1)
  );
  drawChart();
}

function setChartMetric(metric) {
  chartMetric = metric;
  selectedChartIndex = null;
  document.querySelectorAll(".metric-button").forEach((button) => {
    const selected = button.dataset.metric === chartMetric;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  renderChart();
}

function setChartRange(range) {
  chartRange = range === "all" ? "all" : Number(range);
  selectedChartIndex = null;
  document.querySelectorAll(".range-button").forEach((button) => {
    const selected = button.dataset.range === String(range);
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  renderChart();
}

function setLanguage(nextLanguage) {
  language = nextLanguage;
  localStorage.setItem(LANGUAGE_KEY, language);
  applyStaticTranslations();
  updateDynamicContent();
}

function installInteractions() {
  document.querySelectorAll(".language-button").forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.language));
  });
  document.querySelectorAll(".metric-button").forEach((button) => {
    button.addEventListener("click", () => setChartMetric(button.dataset.metric));
  });
  document.querySelectorAll(".range-button").forEach((button) => {
    button.addEventListener("click", () => setChartRange(button.dataset.range));
  });
  document.querySelectorAll("[data-dialog-open]").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById(button.dataset.dialogOpen).showModal();
    });
  });
  document.querySelectorAll("[data-dialog-close]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });

  const canvas = document.getElementById("trendChart");
  canvas.addEventListener("pointerdown", (event) => {
    isDraggingChart = true;
    canvas.setPointerCapture(event.pointerId);
    selectChartPoint(event.clientX);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (isDraggingChart) selectChartPoint(event.clientX);
  });
  canvas.addEventListener("pointerup", () => {
    isDraggingChart = false;
  });
  canvas.addEventListener("pointercancel", () => {
    isDraggingChart = false;
  });
  canvas.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === "ArrowLeft" ? -1 : 1;
    selectedChartIndex = Math.min(
      activeChartRows.length - 1,
      Math.max(0, (selectedChartIndex ?? activeChartRows.length - 1) + direction)
    );
    drawChart();
  });
}

async function initialize() {
  applyStaticTranslations();
  installInteractions();
  try {
    dashboardData = window.DENGUE_DASHBOARD_DATA || null;
    if (window.location.protocol !== "file:") {
      try {
        const response = await fetch(DATA_URL, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Dashboard data returned ${response.status}`);
        }
        dashboardData = await response.json();
      } catch (fetchError) {
        if (!dashboardData) throw fetchError;
        console.warn("Using the bundled dashboard snapshot.", fetchError);
      }
    }
    if (!dashboardData) throw new Error("No dashboard snapshot is available.");
    document.getElementById("loadingState").hidden = true;
    document.getElementById("dashboard").hidden = false;
    updateDynamicContent();
  } catch (error) {
    console.error(error);
    document.getElementById("loadingState").hidden = true;
    document.getElementById("errorState").hidden = false;
  }
}

initialize();
