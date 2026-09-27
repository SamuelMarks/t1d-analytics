/**
 * @file i18n.ts
 * Initializes and exports the i18next instance for internationalization.
 */
import i18next from "i18next";

/**
 * English translation resources.
 */
export const enTranslations = {
  app: {
    title: "t1d-analytics",
    play: "Play",
    chatNumber: "Chat #{{count}}",
    chatNumber_one: "Chat #{{count}}",
    chatNumber_other: "Chat #{{count}}",
    temporaryChat: "Temporary chat",
    copyOf: "{{title}} (Copy)",
    newChat: "+ New Chat",
    schemaExplorer: "Schema Explorer",
    loadingSchema: "Loading schema...",
    model: "Model:",
    database: "Database:",
    gemma4: "Gemma 4",
    literalSql: "Literal SQL",
    emptyState: "Select or create a chat to begin.",
    typeMessage: "Type a message...",
    send: "Send",
    tableData: "Table Data",
    loading: "Loading...",
    previous: "Previous",
    next: "Next",
    page: "Page {{page}}",
  },
  ui: {
    rawSql: "Raw SQL",
    modelName: "Model: {{name}}",
    errorDetails: "Error details: {{error}}",
    errorComm: "Error communicating with backend API: {{error}}",
    failedSchema: "Failed to load schema",
    noTables: "No tables found. Load data first.",
    querying: "Querying...",
    failedData: "Failed to load data: {{error}}",
    newChatTitle: "Enter new chat title:",
    deleteConfirm: 'Are you sure you want to delete "{{title}}"?',
    tableName: "Table: {{name}}",
    rename: "Rename",
    duplicate: "Duplicate",
    delete: "Delete",
    noRows: "No rows returned",
    viewTableData: "View Table Data",
    noDataAvailable: "No data available.",
    noMessagesYet:
      "No messages yet. Send a message to start, or try an example query:",
    exampleQueries: {
      patientDemographics: "[SQL] Patient Demographics",
      adverseEvents: "[SQL] Adverse Events Frequency",
      pumpManufacturers: "[SQL] Pump Manufacturers",
      hba1cDemographics: "[SQL] HbA1c by Demographics",
      nlpFirst5: "[NLP] Show First 5 Patients",
    },
    copyQuery: "Copy Query",
    refreshQuery: "Refresh Query",
    databaseSwitched: "Database switched to {{db}}",
    sessionsSynced: "Chats synced with server",
    settingsSaved: "Provider settings saved",
    settingsCleared: "Provider keys cleared",
  },
  aria: {
    sidebar: "Sidebar",
    startNewChat: "Start a new chat",
    closeSidebar: "Close sidebar",
    chatHistory: "Chat history",
    toggleSchema: "Toggle Schema Explorer",
    mainChatArea: "Main chat area",
    openSidebar: "Open sidebar",
    selectModel: "Select AI Model",
    selectDatabase: "Select Database",
    syncSessions: "Sync Sessions",
    cohortFilter: "Cohort Query Builder",
    toggleTheme: "Toggle dark and light mode",
    switchToLightMode: "Switch to light mode",
    switchToDarkMode: "Switch to dark mode",
    chatInputForm: "Chat input form",
    chatInputField: "Chat input field",
    sendMessage: "Send message",
    closeModal: "Close modal",
    viewTableData: "View Table Data",
    playQuery: "Play Query",
    copyQuery: "Copy Query",
    refreshQuery: "Refresh Query",
    editChatTitle: "Edit chat title",
    duplicateChat: "Duplicate chat",
    deleteChat: "Delete chat",
    chatActions: "Chat actions",
    chatOptions: "Chat options",
    codeBlock: "Code block",
    chatInputHelp: "Press Enter to send, Shift+Enter for a new line",
    languageSelector: "Select application language",
    messageReceived: "Message received",
  },
  cohort: {
    title: "Visual Cohort Query Builder",
    table: "Primary Table:",
    joinTable: "Secondary Table (JOIN):",
    joinType: "Join Type:",
    joinOn: "Join Condition (ON):",
    minAge: "Min Age:",
    maxAge: "Max Age:",
    gender: "Gender:",
    txGroup: "Treatment Group:",
    maxTIR: "Max TIR % (<=):",
    minHbA1c: "Min HbA1c % (>=):",
    sqlPreview: "Generated DuckDB SQL:",
    insertChat: "Insert into Chat",
    execute: "Run Cohort Query",
  },
  provider: {
    title: "Cloud LLM Providers & API Keys",
    description:
      "Configure external cloud LLM providers (OpenAI, Anthropic, Google). Keys are stored locally in your browser.",
    consent: "Save keys in browser localStorage",
    clear: "Clear Stored Keys",
    save: "Save Settings",
  },
  table: {
    searchPlaceholder: "Search table rows...",
    exportCsv: "Export CSV",
    exportJson: "Export JSON",
    fullCsv: "Full CSV",
    fullExcel: "Full Excel",
    fullParquet: "Full Parquet",
    sortDefault: "(Default)",
    sortAsc: "ASC",
    sortDesc: "DESC",
  },
  backend: {
    sqlExecution: "SQL execution error: {{error}}",
    missingSdk: "any-llm-sdk[ollama] is not installed.",
    readSchemaFailed: "Failed to read schema: {{error}}",
    llmTranslationError: "LLM translation error: {{error}}",
    emptyMessage: "Message cannot be empty.",
    literalSql: "Executed literal SQL:",
    generatedSql: "Generated SQL (click 'Run SQL' to execute):",
    errorDbExecution: "Error executing database operation.",
    errorNlpTranslation: "Error during NLP translation.",
    errorUnexpected: "An unexpected error occurred.",
    invalidTable: "Invalid table name",
    tableNotFound: "Table not found",
    serverError: "Internal server error: {{error}}",
    dbNotFound: "Database file not found: {{path}}",
    dbEmpty: "Database contains 0 tables.",
    dbUnreadable: "Database is unreadable or corrupted.",
    invalidDatabasePath: "Invalid or unauthorized database path: {{path}}",
    invalidPagination:
      "Invalid pagination parameters: limit must be positive, offset non-negative.",
  },
  status: {
    offlineTitle: "Backend Server Unreachable",
    offlineDesc:
      "Cannot connect to the backend server at {{url}}. Ensure the API server is running.",
    dbMissingTitle: "Database File Missing",
    dbMissingDesc:
      "Database file not found at '{{path}}'. Run 't1d-analytics load' to initialize it.",
    dbEmptyTitle: "Database Unpopulated",
    dbEmptyDesc:
      "Database contains 0 tables. Load clinical trial data using 't1d-analytics load'.",
    dbMissingDataTitle: "Clinical Datasets Missing",
    dbMissingDataDesc:
      "Standard T1D clinical trial tables (patients, cgms, etc.) are missing.",
    ollamaOfflineTitle: "Ollama LLM Offline",
    ollamaOfflineDesc:
      "Natural language translation is unavailable. Switched to Literal SQL mode.",
    retry: "Retry Connection",
    reconnecting: "Reconnecting...",
    dismiss: "Dismiss alert",
    systemHealthy: "All systems operational",
    systemDegraded: "Degraded service",
    systemError: "System error",
    systemOffline: "Backend offline",
    diagnostics: "Diagnostics",
    setupGuide: "Setup Instructions",
    copyCommand: "Copy Command",
    commandCopied: "Copied!",
    reloadSchema: "Reload Schema",
    inputOffline: "Backend unreachable. Reconnecting...",
    backendStatus: "Backend API",
    databaseStatus: "DuckDB Database",
    llmStatus: "Ollama LLM",
    connected: "Connected",
    disconnected: "Disconnected",
    healthy: "Healthy",
    warning: "Warning",
    error: "Error",
    emptyDbPrompt:
      "Database has no tables loaded. Use 't1d-analytics load' to import clinical data.",
  },
  cgm: {
    title: "📊 CGM Analytics (Ambulatory Glucose Profile & TIR)",
    reportPdf: "🖨️ Clinical AGP Report (PDF)",
    wearValid:
      "✓ Sensor Wear: {{pct}}% ({{days}} days, {{readings}} readings) - Valid (>=70%)",
    wearCaution:
      "⚠️ Sensor Wear: {{pct}}% ({{days}} days) - Caution (<70% wear)",
    gmi: "GMI: {{gmi}}% (Mean: {{mean}} mg/dL)",
    cv: "CV: {{cv}}% (Target ≤36% | SD: {{sd}} mg/dL)",
    lbgiHbgi: "LBGI: {{lbgi}} | HBGI: {{hbgi}}",
    dayNight: "Day: {{day}}% | Night: {{night}}%",
    obsWindow: "Observation Window:",
    days7: "7 Days",
    days14: "14 Days (Standard)",
    days30: "30 Days",
    days90: "90 Days",
    daysAll: "All Available Data",
    timeOfDay: "Time of Day (Hour)",
    glucoseMgDl: "Glucose (mg/dL)",
    targetRange: "Target Range (70-180 mg/dL)",
    agpCurveTitle: "24-Hour Ambulatory Glucose Profile (Median & Percentiles)",
    agpSummaryAria:
      "24-Hour Ambulatory Glucose Profile (AGP) Curve: median glucose across 24 hours with 5th to 95th percentile ranges and 70-180 mg/dL target band ({{readings}} readings across {{days}} days).",
    tirChartAria:
      "Time in Range Chart: In Range {{inRange}}%, Low {{low}}%, Very Low {{veryLow}}%, High {{high}}%, Very High {{veryHigh}}%",
    histChartAria: "Glucose Distribution Histogram",
    veryLowLabel: "Very Low (<54)",
    lowLabel: "Low (54-69)",
    inRangeLabel: "In Range (70-180)",
    highLabel: "High (181-250)",
    veryHighLabel: "Very High (>250)",
    meanLabel:
      "Mean: {{mean}} mg/dL | Readings: {{readings}} | Target (70-180 mg/dL): {{inRange}}%",
  },
};

/**
 * Japanese translation resources.
 */
export const jaTranslations = {
  app: {
    title: "t1d-analytics",
    play: "実行",
    chatNumber: "チャット #{{count}}",
    chatNumber_other: "チャット #{{count}}",
    temporaryChat: "一時的なチャット",
    copyOf: "{{title}} (コピー)",
    newChat: "+ 新しいチャット",
    schemaExplorer: "スキーマエクスプローラー",
    loadingSchema: "スキーマを読み込み中...",
    model: "モデル:",
    database: "データベース:",
    gemma4: "Gemma 4",
    literalSql: "リテラルSQL",
    emptyState: "チャットを選択または作成して開始してください。",
    typeMessage: "メッセージを入力...",
    send: "送信",
    tableData: "テーブルデータ",
    loading: "読み込み中...",
    previous: "前へ",
    next: "次へ",
    page: "ページ {{page}}",
  },
  ui: {
    rawSql: "生SQL",
    modelName: "モデル: {{name}}",
    errorDetails: "エラー詳細: {{error}}",
    errorComm: "バックエンドAPIとの通信エラー: {{error}}",
    failedSchema: "スキーマの読み込みに失敗しました",
    noTables: "テーブルが見つかりません。まずデータをロードしてください。",
    querying: "クエリ実行中...",
    failedData: "データの読み込みに失敗しました: {{error}}",
    newChatTitle: "新しいチャットのタイトルを入力:",
    deleteConfirm: '本当に "{{title}}" を削除しますか？',
    tableName: "テーブル: {{name}}",
    rename: "名前の変更",
    duplicate: "複製",
    delete: "削除",
    noRows: "行が返されませんでした",
    viewTableData: "テーブルデータを表示",
    noDataAvailable: "データがありません。",
    noMessagesYet:
      "メッセージはまだありません。メッセージを送信して開始するか、サンプルのクエリを試してください：",
    exampleQueries: {
      patientDemographics: "[SQL] 患者の人口統計",
      adverseEvents: "[SQL] 有害事象の頻度",
      pumpManufacturers: "[SQL] ポンプメーカー",
      hba1cDemographics: "[SQL] 人口統計別のHbA1c",
      nlpFirst5: "[NLP] 最初の5人の患者を表示",
    },
    copyQuery: "クエリをコピー",
    refreshQuery: "クエリを更新",
    databaseSwitched: "データベースが {{db}} に切り替わりました",
    sessionsSynced: "チャットがサーバーと同期されました",
    settingsSaved: "プロバイダー設定が保存されました",
    settingsCleared: "プロバイダーキーがクリアされました",
  },
  aria: {
    sidebar: "サイドバー",
    startNewChat: "新しいチャットを開始",
    closeSidebar: "サイドバーを閉じる",
    chatHistory: "チャット履歴",
    toggleSchema: "スキーマエクスプローラーの切り替え",
    mainChatArea: "メインチャットエリア",
    openSidebar: "サイドバーを開く",
    selectModel: "AIモデルの選択",
    selectDatabase: "データベースを選択",
    syncSessions: "セッションを同期",
    cohortFilter: "コホートクエリビルダー",
    toggleTheme: "ダークモードとライトモードの切り替え",
    switchToLightMode: "ライトモードに切り替え",
    switchToDarkMode: "ダークモードに切り替え",
    chatInputForm: "チャット入力フォーム",
    chatInputField: "チャット入力フィールド",
    sendMessage: "メッセージを送信",
    closeModal: "モーダルを閉じる",
    viewTableData: "テーブルデータを表示",
    playQuery: "クエリを実行",
    copyQuery: "クエリをコピー",
    refreshQuery: "クエリを更新",
    editChatTitle: "チャットタイトルを編集",
    duplicateChat: "チャットを複製",
    deleteChat: "チャットを削除",
    chatActions: "チャットアクション",
    chatOptions: "チャットオプション",
    codeBlock: "コードブロック",
    chatInputHelp: "Enterで送信、Shift+Enterで改行",
    languageSelector: "アプリケーション言語を選択",
    messageReceived: "メッセージを受信しました",
  },
  cohort: {
    title: "コホートクエリビルダー",
    table: "プライマリテーブル:",
    joinTable: "セカンダリテーブル (JOIN):",
    joinType: "結合タイプ:",
    joinOn: "結合条件 (ON):",
    minAge: "最小年齢:",
    maxAge: "最大年齢:",
    gender: "性別:",
    txGroup: "治療グループ:",
    maxTIR: "最大TIR % (<=):",
    minHbA1c: "最小HbA1c % (>=):",
    sqlPreview: "生成されたDuckDB SQL:",
    insertChat: "チャットに挿入",
    execute: "コホートクエリを実行",
  },
  provider: {
    title: "クラウドLLMプロバイダーとAPIキー",
    description:
      "外部クラウドLLMプロバイダー（OpenAI、Anthropic、Google）を設定します。キーはブラウザにローカル保存されます。",
    consent: "ブラウザのlocalStorageにキーを保存する",
    clear: "保存されたキーをクリア",
    save: "設定を保存",
  },
  table: {
    searchPlaceholder: "テーブル行を検索...",
    exportCsv: "CSVエクスポート",
    exportJson: "JSONエクスポート",
    fullCsv: "全行CSV",
    fullExcel: "全行Excel",
    fullParquet: "全行Parquet",
    sortDefault: "(デフォルト)",
    sortAsc: "昇順",
    sortDesc: "降順",
  },
  backend: {
    sqlExecution: "SQL実行エラー: {{error}}",
    missingSdk: "any-llm-sdk[ollama] がインストールされていません。",
    readSchemaFailed: "スキーマの読み込みに失敗しました: {{error}}",
    llmTranslationError: "LLM翻訳エラー: {{error}}",
    emptyMessage: "メッセージを空にすることはできません。",
    literalSql: "実行されたリテラルSQL:",
    generatedSql: "生成されたSQL ('Run SQL' をクリックして実行):",
    errorDbExecution: "データベース操作の実行中にエラーが発生しました。",
    errorNlpTranslation: "NLP翻訳中のエラー。",
    errorUnexpected: "予期しないエラーが発生しました。",
    invalidTable: "無効なテーブル名",
    tableNotFound: "テーブルが見つかりません",
    serverError: "内部サーバーエラー: {{error}}",
    dbNotFound: "データベースファイルが見つかりません: {{path}}",
    dbEmpty: "データベースにテーブルがありません。",
    dbUnreadable: "データベースを読み取れないか破損しています。",
    invalidDatabasePath: "無効または未承認のデータベースパス: {{path}}",
    invalidPagination:
      "無効なページネーションパラメータ: limitは正数、offsetは0以上である必要があります。",
  },
  status: {
    offlineTitle: "バックエンドサーバーに接続できません",
    offlineDesc:
      "{{url}} のバックエンドサーバーに接続できません。APIサーバーが実行中であることを確認してください。",
    dbMissingTitle: "データベースファイルが見つかりません",
    dbMissingDesc:
      "データベースファイル '{{path}}' が見つかりません。't1d-analytics load' を実行して初期化してください。",
    dbEmptyTitle: "データベースが空です",
    dbEmptyDesc:
      "データベースにテーブルがありません。't1d-analytics load' で臨床試験データを読み込んでください。",
    dbMissingDataTitle: "臨床データセットが不足しています",
    dbMissingDataDesc:
      "標準的なT1D臨床試験テーブル（patients、cgms等）が存在しません。",
    ollamaOfflineTitle: "Ollama LLM オフライン",
    ollamaOfflineDesc:
      "自然言語の翻訳が利用できません。リテラルSQLモードに切り替えました。",
    retry: "再接続を試みる",
    reconnecting: "再接続中...",
    dismiss: "アラートを閉じる",
    systemHealthy: "すべてのシステムが正常に動作中",
    systemDegraded: "一部の機能が制限されています",
    systemError: "システムエラー",
    systemOffline: "バックエンドオフライン",
    diagnostics: "診断情報",
    setupGuide: "セットアップ手順",
    copyCommand: "コマンドをコピー",
    commandCopied: "コピーしました！",
    reloadSchema: "スキーマを再読み込み",
    inputOffline: "バックエンドに接続できません。再接続中...",
    backendStatus: "バックエンドAPI",
    databaseStatus: "DuckDBデータベース",
    llmStatus: "Ollama LLM",
    connected: "接続済み",
    disconnected: "未接続",
    healthy: "正常",
    warning: "警告",
    error: "エラー",
    emptyDbPrompt:
      "データベースにテーブルが読み込まれていません。't1d-analytics load' を使用してデータをインポートしてください。",
  },
  cgm: {
    title: "📊 CGM分析（外来血糖プロファイル＆TIR）",
    reportPdf: "🖨️ 臨床AGPレポート（PDF）",
    wearValid:
      "✓ センサー装着率: {{pct}}% ({{days}}日間、{{readings}}測定) - 有効 (>=70%)",
    wearCaution: "⚠️ センサー装着率: {{pct}}% ({{days}}日間) - 注意 (<70%装着)",
    gmi: "GMI: {{gmi}}% (平均: {{mean}} mg/dL)",
    cv: "CV: {{cv}}% (目標 ≤36% | SD: {{sd}} mg/dL)",
    lbgiHbgi: "LBGI: {{lbgi}} | HBGI: {{hbgi}}",
    dayNight: "昼間: {{day}}% | 夜間: {{night}}%",
    obsWindow: "観察期間:",
    days7: "7日間",
    days14: "14日間（標準）",
    days30: "30日間",
    days90: "90日間",
    daysAll: "すべての利用可能なデータ",
    timeOfDay: "時刻 (時)",
    glucoseMgDl: "血糖値 (mg/dL)",
    targetRange: "目標範囲 (70-180 mg/dL)",
    agpCurveTitle: "24時間外来血糖プロファイル（中央値およびパーセンタイル）",
    agpSummaryAria:
      "24時間外来血糖プロファイル（AGP）曲線: 24時間の中央値血糖、第5〜第95パーセンタイル範囲、70-180 mg/dL目標範囲（{{days}}日間にわたる{{readings}}件の測定値）。",
    tirChartAria:
      "Time in Range チャート: 目標範囲内 {{inRange}}%、低血糖 {{low}}%、超低血糖 {{veryLow}}%、高血糖 {{high}}%、超高血糖 {{veryHigh}}%",
    histChartAria: "血糖値分布ヒストグラム",
    veryLowLabel: "超低血糖 (<54)",
    lowLabel: "低血糖 (54-69)",
    inRangeLabel: "目標範囲内 (70-180)",
    highLabel: "高血糖 (181-250)",
    veryHighLabel: "超高血糖 (>250)",
    meanLabel:
      "平均: {{mean}} mg/dL | 測定数: {{readings}} | 目標 (70-180 mg/dL): {{inRange}}%",
  },
};

/**
 * Arabic translation resources.
 */
export const arTranslations = {
  app: {
    title: "t1d-analytics",
    play: "تشغيل",
    chatNumber: "الدردشة #{{count}}",
    chatNumber_zero: "لا توجد محادثات",
    chatNumber_one: "محادثة واحدة ({{count}})",
    chatNumber_two: "محادثتان ({{count}})",
    chatNumber_few: "{{count}} محادثات",
    chatNumber_many: "{{count}} محادثة",
    chatNumber_other: "{{count}} محادثة",
    temporaryChat: "دردشة مؤقتة",
    copyOf: "{{title}} (نسخة)",
    newChat: "+ دردشة جديدة",
    schemaExplorer: "مستكشف المخطط",
    loadingSchema: "جاري تحميل المخطط...",
    model: "النموذج:",
    database: "قاعدة البيانات:",
    gemma4: "Gemma 4",
    literalSql: "SQL حرفي",
    emptyState: "حدد أو أنشئ دردشة للبدء.",
    typeMessage: "اكتب رسالة...",
    send: "إرسال",
    tableData: "بيانات الجدول",
    loading: "جاري التحميل...",
    previous: "السابق",
    next: "التالي",
    page: "الصفحة {{page}}",
  },
  ui: {
    rawSql: "SQL خام",
    modelName: "النموذج: {{name}}",
    errorDetails: "تفاصيل الخطأ: {{error}}",
    errorComm: "خطأ في الاتصال بواجهة برمجة تطبيقات الواجهة الخلفية: {{error}}",
    failedSchema: "فشل تحميل المخطط",
    noTables: "لم يتم العثور على جداول. قم بتحميل البيانات أولاً.",
    querying: "جاري الاستعلام...",
    failedData: "فشل تحميل البيانات: {{error}}",
    newChatTitle: "أدخل عنوان الدردشة الجديد:",
    deleteConfirm: 'هل أنت متأكد أنك تريد حذف "{{title}}"؟',
    tableName: "الجدول: {{name}}",
    rename: "إعادة تسمية",
    duplicate: "تكرار",
    delete: "حذف",
    noRows: "لم يتم إرجاع أي صفوف",
    viewTableData: "عرض بيانات الجدول",
    noDataAvailable: "لا توجد بيانات متاحة.",
    noMessagesYet:
      "لا توجد رسائل بعد. أرسل رسالة للبدء، أو جرب استعلاماً مثالياً:",
    exampleQueries: {
      patientDemographics: "[SQL] التركيبة السكانية للمرضى",
      adverseEvents: "[SQL] تكرار الأحداث السلبية",
      pumpManufacturers: "[SQL] الشركات المصنعة للمضخات",
      hba1cDemographics: "[SQL] نسبة السكر التراكمي حسب التركيبة السكانية",
      nlpFirst5: "[NLP] عرض أول 5 مرضى",
    },
    copyQuery: "نسخ الاستعلام",
    refreshQuery: "تحديث الاستعلام",
    databaseSwitched: "تم تبديل قاعدة البيانات إلى {{db}}",
    sessionsSynced: "تمت مزامنة الدردشات مع الخادم",
    settingsSaved: "تم حفظ إعدادات الموفر",
    settingsCleared: "تم مسح مفاتيح الموفر",
  },
  aria: {
    sidebar: "الشريط الجانبي",
    startNewChat: "بدء دردشة جديدة",
    closeSidebar: "إغلاق الشريط الجانبي",
    chatHistory: "سجل الدردشة",
    toggleSchema: "تبديل مستكشف المخطط",
    mainChatArea: "منطقة الدردشة الرئيسية",
    openSidebar: "فتح الشريط الجانبي",
    selectModel: "تحديد نموذج الذكاء الاصطناعي",
    selectDatabase: "اختر قاعدة البيانات",
    syncSessions: "مزامنة الجلسات",
    cohortFilter: "منشئ استعلامات المجموعات",
    toggleTheme: "تبديل الوضع الداكن والفاتح",
    switchToLightMode: "التبديل إلى الوضع الفاتح",
    switchToDarkMode: "التبديل إلى الوضع الداكن",
    chatInputForm: "نموذج إدخال الدردشة",
    chatInputField: "حقل إدخال الدردشة",
    sendMessage: "إرسال رسالة",
    closeModal: "إغلاق النافذة المنبثقة",
    viewTableData: "عرض بيانات الجدول",
    playQuery: "تشغيل الاستعلام",
    copyQuery: "نسخ الاستعلام",
    refreshQuery: "تحديث الاستعلام",
    editChatTitle: "تعديل عنوان الدردشة",
    duplicateChat: "تكرار الدردشة",
    deleteChat: "حذف الدردشة",
    chatActions: "إجراءات الدردشة",
    chatOptions: "خيارات الدردشة",
    codeBlock: "كتلة التعليمات البرمجية",
    chatInputHelp: "اضغط على Enter للإرسال، Shift+Enter لسطر جديد",
    languageSelector: "تحديد لغة التطبيق",
    messageReceived: "تم استلام الرسالة",
  },
  cohort: {
    title: "منشئ استعلامات المجموعات",
    table: "الجدول الأساسي:",
    joinTable: "الجدول الثانوي (JOIN):",
    joinType: "نوع الربط:",
    joinOn: "شرط الربط (ON):",
    minAge: "الحد الأدنى للعمر:",
    maxAge: "الحد الأقصى للعمر:",
    gender: "الجنس:",
    txGroup: "مجموعة العلاج:",
    maxTIR: "الحد الأقصى لـ TIR % (<=):",
    minHbA1c: "الحد الأدنى لـ HbA1c % (>=):",
    sqlPreview: "DuckDB SQL الذي تم إنشاؤه:",
    insertChat: "إدراج في الدردشة",
    execute: "تشغيل استعلام المجموعة",
  },
  provider: {
    title: "موفرو خدمة LLM السحابية ومفاتيح API",
    description:
      "تكوين موفري LLM السحابيين الخارجيين (OpenAI, Anthropic, Google). يتم تخزين المفاتيح محليًا في متصفحك.",
    consent: "حفظ المفاتيح في localStorage بالمتصفح",
    clear: "مسح المفاتيح المخزنة",
    save: "حفظ الإعدادات",
  },
  table: {
    searchPlaceholder: "البحث في صفوف الجدول...",
    exportCsv: "تصدير CSV",
    exportJson: "تصدير JSON",
    fullCsv: "CSV كامل",
    fullExcel: "Excel كامل",
    fullParquet: "Parquet كامل",
    sortDefault: "(افتراضي)",
    sortAsc: "تصاعدي",
    sortDesc: "تنازلي",
  },
  backend: {
    sqlExecution: "خطأ في تنفيذ SQL: {{error}}",
    missingSdk: "حزمة any-llm-sdk[ollama] غير مثبتة.",
    readSchemaFailed: "فشل في قراءة المخطط: {{error}}",
    llmTranslationError: "خطأ في ترجمة LLM: {{error}}",
    emptyMessage: "لا يمكن أن تكون الرسالة فارغة.",
    literalSql: "تم تنفيذ SQL حرفي:",
    generatedSql: "SQL مُنشأ (انقر 'Run SQL' للتنفيذ):",
    errorDbExecution: "خطأ في تنفيذ عملية قاعدة البيانات.",
    errorNlpTranslation: "خطأ أثناء ترجمة البرمجة اللغوية العصبية.",
    errorUnexpected: "حدث خطأ غير متوقع.",
    invalidTable: "اسم جدول غير صالح",
    tableNotFound: "الجدول غير موجود",
    serverError: "خطأ خادم داخلي: {{error}}",
    dbNotFound: "لم يتم العثور على ملف قاعدة البيانات: {{path}}",
    dbEmpty: "قاعدة البيانات لا تحتوي على جداول.",
    dbUnreadable: "قاعدة البيانات غير قابلة للقراءة أو تالفة.",
    invalidDatabasePath: "مسار قاعدة بيانات غير صالح أو غير مصرح به: {{path}}",
    invalidPagination:
      "معلمات ترقيم الصفحات غير صالحة: يجب أن يكون الحد موجبًا والإزاحة غير سالبة.",
  },
  status: {
    offlineTitle: "تعذر الاتصال بخادم الواجهة الخلفية",
    offlineDesc:
      "لا يمكن الاتصال بخادم الواجهة الخلفية على {{url}}. تأكد من تشغيل خادم API.",
    dbMissingTitle: "ملف قاعدة البيانات مفقود",
    dbMissingDesc:
      "لم يتم العثور على ملف قاعدة البيانات في '{{path}}'. شغّل 't1d-analytics load' لتهيئته.",
    dbEmptyTitle: "قاعدة البيانات غير مأهولة",
    dbEmptyDesc:
      "قاعدة البيانات لا تحتوي على جداول. حمّل بيانات التجارب السريرية باستخدام 't1d-analytics load'.",
    dbMissingDataTitle: "مجموعات البيانات السريرية مفقودة",
    dbMissingDataDesc:
      "جداول التجارب السريرية القياسية للنوع الأول من السكري (patients, cgms, إلخ) مفقودة.",
    ollamaOfflineTitle: "خدمة Ollama LLM غير متصلة",
    ollamaOfflineDesc:
      "ترجمة استعلامات اللغة الطبيعية غير متاحة. تم التبديل إلى وضع SQL الحرفي.",
    retry: "إعادة المحاولة",
    reconnecting: "جاري إعادة الاتصال...",
    dismiss: "إغلاق التنبيه",
    systemHealthy: "جميع الأنظمة تعمل بشكل طبيعي",
    systemDegraded: "الخدمة متدهورة جزئياً",
    systemError: "خطأ في النظام",
    systemOffline: "الواجهة الخلفية غير متصلة",
    diagnostics: "التشخيصات",
    setupGuide: "إرشادات الإعداد",
    copyCommand: "نسخ الأمر",
    commandCopied: "تم النسخ!",
    reloadSchema: "إعادة تحميل المخطط",
    inputOffline: "تعذر الاتصال بالواجهة الخلفية. جاري إعادة الاتصال...",
    backendStatus: "واجهة برمجة تطبيقات الواجهة الخلفية",
    databaseStatus: "قاعدة بيانات DuckDB",
    llmStatus: "Ollama LLM",
    connected: "متصل",
    disconnected: "غير متصل",
    healthy: "سليم",
    warning: "تحذير",
    error: "خطأ",
    emptyDbPrompt:
      "لا توجد جداول محملة في قاعدة البيانات. استخدم 't1d-analytics load' لاستيراد البيانات.",
  },
  cgm: {
    title: "📊 تحليلات CGM (مخطط الجلوكوز المتنقل وTIR)",
    reportPdf: "🖨️ تقرير AGP السريري (PDF)",
    wearValid:
      "✓ ارتداء المستشعر: {{pct}}% ({{days}} يوم، {{readings}} قراءة) - صالح (>=70%)",
    wearCaution:
      "⚠️ ارتداء المستشعر: {{pct}}% ({{days}} يوم) - تحذير (<70% ارتداء)",
    gmi: "GMI: {{gmi}}% (المتوسط: {{mean}} ملغ/ديسيلتر)",
    cv: "CV: {{cv}}% (الهدف ≤36% | الانحراف المعياري: {{sd}} ملغ/ديسيلتر)",
    lbgiHbgi: "LBGI: {{lbgi}} | HBGI: {{hbgi}}",
    dayNight: "نهاراً: {{day}}% | ليلاً: {{night}}%",
    obsWindow: "نافذة الملاحظة:",
    days7: "7 أيام",
    days14: "14 يوم (قياسي)",
    days30: "30 يوم",
    days90: "90 يوم",
    daysAll: "جميع البيانات المتاحة",
    timeOfDay: "الوقت من اليوم (ساعة)",
    glucoseMgDl: "الجلوكوز (ملغ/ديسيلتر)",
    targetRange: "النطاق المستهدف (70-180 ملغ/ديسيلتر)",
    agpCurveTitle:
      "مخطط الجلوكوز المتنقل على مدار 24 ساعة (الوسيط والنسب المئوية)",
    agpSummaryAria:
      "منحنى ملف تعريف الجلوكوز الإسعافي (AGP) على مدار 24 ساعة: متوسط الجلوكوز على مدار 24 ساعة مع نطاقات النسبة المئوية من 5 إلى 95 ونطاق مستهدف من 70-180 ملغ/ديسيلتر ({{readings}} قراءة عبر {{days}} يومًا).",
    tirChartAria:
      "مخطط الوقت في النطاق: في النطاق {{inRange}}%، منخفض {{low}}%، منخفض جدًا {{veryLow}}%، مرتفع {{high}}%، مرتفع جدًا {{veryHigh}}%",
    histChartAria: "مدرج تكراري لتوزيع الجلوكوز",
    veryLowLabel: "منخفض جداً (<54)",
    lowLabel: "منخفض (54-69)",
    inRangeLabel: "في النطاق (70-180)",
    highLabel: "مرتفع (181-250)",
    veryHighLabel: "مرتفع جداً (>250)",
    meanLabel:
      "المتوسط: {{mean}} ملغ/ديسيلتر | القراءات: {{readings}} | الهدف (70-180 ملغ/ديسيلتر): {{inRange}}%",
  },
};

/**
 * Hebrew translation resources.
 */
export const heTranslations = {
  app: {
    title: "t1d-analytics",
    play: "הפעל",
    chatNumber: "צ'אט #{{count}}",
    chatNumber_one: "צ'אט #{{count}}",
    chatNumber_two: "2 צ'אטים",
    chatNumber_many: "{{count}} צ'אטים",
    chatNumber_other: "{{count}} צ'אטים",
    temporaryChat: "צ'אט זמני",
    copyOf: "{{title}} (עותק)",
    newChat: "+ צ'אט חדש",
    schemaExplorer: "סייר סכמות",
    loadingSchema: "טוען סכמה...",
    model: "מודל:",
    database: "מסד נתונים:",
    gemma4: "Gemma 4",
    literalSql: "SQL מילולי",
    emptyState: "בחר או צור צ'אט כדי להתחיל.",
    typeMessage: "הקלד הודעה...",
    send: "שלח",
    tableData: "נתוני טבלה",
    loading: "טוען...",
    previous: "הקודם",
    next: "הבא",
    page: "עמוד {{page}}",
  },
  ui: {
    rawSql: "SQL גולמי",
    modelName: "מודל: {{name}}",
    errorDetails: "פרטי שגיאה: {{error}}",
    errorComm: "שגיאה בתקשורת עם ה-API: {{error}}",
    failedSchema: "טעינת סכמה נכשלה",
    noTables: "לא נמצאו טבלאות. טען נתונים תחילה.",
    querying: "מתשאל...",
    failedData: "טעינת נתונים נכשלה: {{error}}",
    newChatTitle: "הזן כותרת חדשה לצ'אט:",
    deleteConfirm: 'האם אתה בטוח שברצונך למחוק את "{{title}}"?',
    tableName: "טבלה: {{name}}",
    rename: "שנה שם",
    duplicate: "שכפל",
    delete: "מחק",
    noRows: "לא הוחזרו שורות",
    viewTableData: "הצג נתוני טבלה",
    noDataAvailable: "אין נתונים זמינים.",
    noMessagesYet:
      "אין הודעות עדיין. שלח הודעה כדי להתחיל, או נסה שאילתה לדוגמה:",
    exampleQueries: {
      patientDemographics: "[SQL] דמוגרפיה של מטופלים",
      adverseEvents: "[SQL] תדירות תופעות לוואי",
      pumpManufacturers: "[SQL] יצרני משאבות",
      hba1cDemographics: "[SQL] HbA1c לפי דמוגרפיה",
      nlpFirst5: "[NLP] הצג את 5 המטופלים הראשונים",
    },
    copyQuery: "העתק שאילתה",
    refreshQuery: "רענן שאילתה",
    databaseSwitched: "מסד הנתונים הוחלף ל-{{db}}",
    sessionsSynced: "הצ'אטים סונכרנו עם השרת",
    settingsSaved: "הגדרות הספק נשמרו",
    settingsCleared: "מפתחות הספק נוקו",
  },
  aria: {
    sidebar: "סרגל צד",
    startNewChat: "התחל צ'אט חדש",
    closeSidebar: "סגור סרגל צד",
    chatHistory: "היסטוריית צ'אט",
    toggleSchema: "החלף סייר סכמות",
    mainChatArea: "אזור צ'אט ראשי",
    openSidebar: "פתח סרגל צד",
    selectModel: "בחר מודל AI",
    selectDatabase: "בחר מסד נתונים",
    syncSessions: "סנכרן הפעלות",
    cohortFilter: "בונה שאילתות קוהורט",
    toggleTheme: "החלף מצב כהה ובהיר",
    switchToLightMode: "עבור למצב בהיר",
    switchToDarkMode: "עבור למצב כהה",
    chatInputForm: "טופס קלט צ'אט",
    chatInputField: "שדה קלט צ'אט",
    sendMessage: "שלח הודעה",
    closeModal: "סגור חלון קופץ",
    viewTableData: "הצג נתוני טבלה",
    playQuery: "הפעל שאילתה",
    copyQuery: "העתק שאילתה",
    refreshQuery: "רענן שאילתה",
    editChatTitle: "ערוך כותרת צ'אט",
    duplicateChat: "שכפל צ'אט",
    deleteChat: "מחק צ'אט",
    chatActions: "פעולות צ'אט",
    chatOptions: "אפשרויות צ'אט",
    codeBlock: "בלוק קוד",
    chatInputHelp: "לחץ Enter לשליחה, Shift+Enter לשורה חדשה",
    languageSelector: "בחר שפת יישום",
    messageReceived: "הודעה התקבלה",
  },
  cohort: {
    title: "בונה שאילתות קוהורט",
    table: "טבלה ראשית:",
    joinTable: "טבלה משנית (JOIN):",
    joinType: "סוג הצטרפות:",
    joinOn: "תנאי הצטרפות (ON):",
    minAge: "גיל מינימלי:",
    maxAge: "גיל מקסימלי:",
    gender: "מגדר:",
    txGroup: "קבוצת טיפול:",
    maxTIR: "TIR מקסימלי % (<=):",
    minHbA1c: "HbA1c מינימלי % (>=):",
    sqlPreview: "DuckDB SQL שנוצר:",
    insertChat: "הכנס לצ'אט",
    execute: "הפעל שאילתת קוהורט",
  },
  provider: {
    title: "ספקי LLM בענן ומפתחות API",
    description:
      "הגדר ספקי LLM חיצוניים בענן (OpenAI, Anthropic, Google). המפתחות נשמרים מקומית בדפדפן שלך.",
    consent: "שמור מפתחות ב-localStorage של הדפדפן",
    clear: "נקה מפתחות שמורים",
    save: "שמור הגדרות",
  },
  table: {
    searchPlaceholder: "חפש בשורות הטבלה...",
    exportCsv: "ייצוא CSV",
    exportJson: "ייצוא JSON",
    fullCsv: "CSV מלא",
    fullExcel: "Excel מלא",
    fullParquet: "Parquet מלא",
    sortDefault: "(ברירת מחדל)",
    sortAsc: "עולה",
    sortDesc: "יורד",
  },
  backend: {
    sqlExecution: "שגיאת ביצוע SQL: {{error}}",
    missingSdk: "any-llm-sdk[ollama] אינו מותקן.",
    readSchemaFailed: "קריאת סכמה נכשלה: {{error}}",
    llmTranslationError: "שגיאת תרגום LLM: {{error}}",
    emptyMessage: "ההודעה אינה יכולה להיות ריקה.",
    literalSql: "בוצע SQL מילולי:",
    generatedSql: "SQL נוצר (לחץ 'Run SQL' לביצוע):",
    errorDbExecution: "שגיאה בביצוע פעולת מסד הנתונים.",
    errorNlpTranslation: "שגיאה במהלך תרגום NLP.",
    errorUnexpected: "אירעה שגיאה בלתי צפויה.",
    invalidTable: "שם טבלה לא חוקי",
    tableNotFound: "טבלה לא נמצאה",
    serverError: "שגיאת שרת פנימית: {{error}}",
    dbNotFound: "קובץ מסד הנתונים לא נמצא: {{path}}",
    dbEmpty: "מסד הנתונים אינו מכיל טבלאות.",
    dbUnreadable: "מסד הנתונים אינו קריא או פגום.",
    invalidDatabasePath: "נתיב מסד נתונים לא חוקי או לא מורשה: {{path}}",
    invalidPagination:
      "פרמטרי דפדוף לא חוקיים: המגבלה חייבת להיות חיובית וההיסט אינו שלילי.",
  },
  status: {
    offlineTitle: "שרת ה-Backend אינו נגיש",
    offlineDesc:
      "לא ניתן להתחבר לשרת ה-Backend ב-{{url}}. ודא ששרת ה-API פעיל.",
    dbMissingTitle: "קובץ מסד הנתונים חסר",
    dbMissingDesc:
      "קובץ מסד הנתונים לא נמצא ב-'{{path}}'. הפעל 't1d-analytics load' כדי לאתחל אותו.",
    dbEmptyTitle: "מסד הנתונים אינו מאוכלס",
    dbEmptyDesc:
      "מסד הנתונים אינו מכיל טבלאות. טען נתוני ניסויים קליניים באמצעות 't1d-analytics load'.",
    dbMissingDataTitle: "ערכות נתונים קליניות חסרות",
    dbMissingDataDesc:
      "טבלאות ניסויים קליניים סטנדרטיים של סוכרת מסוג 1 (patients, cgms וכו') חסרות.",
    ollamaOfflineTitle: "שירות Ollama LLM לא מקוון",
    ollamaOfflineDesc:
      "תרגום שאילתות בשפה טבעית אינו זמין. בוצע מעבר למצב SQL מילולי.",
    retry: "נסה שוב",
    reconnecting: "מתחבר מחדש...",
    dismiss: "סגור התראה",
    systemHealthy: "כל המערכות פועלות כסדרן",
    systemDegraded: "שירות מוגבל",
    systemError: "שגיאת מערכת",
    systemOffline: "Backend לא מקוון",
    diagnostics: "אבחון מערכת",
    setupGuide: "הוראות הגדרה",
    copyCommand: "העתק פקודה",
    commandCopied: "הועתק!",
    reloadSchema: "טען מחדש סכמה",
    inputOffline: "אין חיבור ל-Backend. מנסה להתחבר מחדש...",
    backendStatus: "ממשק Backend API",
    databaseStatus: "מסד נתונים DuckDB",
    llmStatus: "Ollama LLM",
    connected: "מחובר",
    disconnected: "מנותק",
    healthy: "תקין",
    warning: "אזהרה",
    error: "שגיאה",
    emptyDbPrompt:
      "אין טבלאות במסד הנתונים. השתמש ב-'t1d-analytics load' לייבוא נתונים.",
  },
  cgm: {
    title: "📊 ניתוח CGM (פרופיל גלוקוז אמבולטורי ו-TIR)",
    reportPdf: "🖨️ דוח AGP קליני (PDF)",
    wearValid:
      "✓ שימוש בחיישן: {{pct}}% ({{days}} ימים, {{readings}} קריאות) - תקין (>=70%)",
    wearCaution:
      "⚠️ שימוש בחיישן: {{pct}}% ({{days}} ימים) - זהירות (<70% שימוש)",
    gmi: 'GMI: {{gmi}}% (ממוצע: {{mean}} מ"ג/דצ"ל)',
    cv: 'CV: {{cv}}% (יעד ≤36% | סטיית תקן: {{sd}} מ"ג/דצ"ל)',
    lbgiHbgi: "LBGI: {{lbgi}} | HBGI: {{hbgi}}",
    dayNight: "יום: {{day}}% | לילה: {{night}}%",
    obsWindow: "חלון תצפית:",
    days7: "7 ימים",
    days14: "14 ימים (סטנדרטי)",
    days30: "30 ימים",
    days90: "90 ימים",
    daysAll: "כל הנתונים הזמינים",
    timeOfDay: "שעה ביום",
    glucoseMgDl: 'גלוקוז (מ"ג/דצ"ל)',
    targetRange: 'טווח יעד (70-180 מ"ג/דצ"ל)',
    agpCurveTitle: "פרופיל גלוקוז אמבולטורי 24 שעות (חציון ואחוזונים)",
    agpSummaryAria:
      'עקומת פרופיל גלוקוז אמבולטורי (AGP) של 24 שעות: חציון גלוקוז על פני 24 שעות עם טווחי אחוזון 5 עד 95 ורצועת יעד של 70-180 מ"ג/דצ"ל ({{readings}} קריאות לאורך {{days}} ימים).',
    tirChartAria:
      "תרשים זמן בטווח: בטווח {{inRange}}%, נמוך {{low}}%, נמוך מאוד {{veryLow}}%, גבוה {{high}}%, גבוה מאוד {{veryHigh}}%",
    histChartAria: "היסטוגרמת התפלגות גלוקוז",
    veryLowLabel: "נמוך מאוד (<54)",
    lowLabel: "נמוך (54-69)",
    inRangeLabel: "בטווח (70-180)",
    highLabel: "גבוה (181-250)",
    veryHighLabel: "גבוה מאוד (>250)",
    meanLabel:
      'ממוצע: {{mean}} מ"ג/דצ"ל | קריאות: {{readings}} | יעד (70-180 מ"ג/דצ"ל): {{inRange}}%',
  },
};

/**
 * Determines initial application language based on storage and navigator settings.
 * @returns {string} The resolved ISO 639-1 language code.
 */
export function getInitialLang(): string {
  const savedLang =
    typeof localStorage !== "undefined"
      ? localStorage.getItem("app-lang")
      : null;
  const browserLang =
    typeof navigator !== "undefined" ? navigator.language.split("-")[0] : "en";
  return (
    savedLang ||
    (["en", "ja", "ar", "he"].includes(browserLang) ? browserLang : "en")
  );
}

const initialLang = getInitialLang();

await i18next.init({
  lng: initialLang,
  fallbackLng: "en",
  resources: {
    en: {
      translation: enTranslations,
    },
    ja: {
      translation: jaTranslations,
    },
    ar: {
      translation: arTranslations,
    },
    he: {
      translation: heTranslations,
    },
  },
});

document.documentElement.lang = initialLang;

/**
 * Sets document direction attribute to 'rtl' or 'ltr' based on language code.
 * @param {string} lang The language code to check.
 */
export function setDocumentDir(lang: string): void {
  document.documentElement.dir = ["ar", "he"].includes(lang) ? "rtl" : "ltr";
}
setDocumentDir(initialLang);

/**
 * Translates HTML elements in the document that have data-i18n attributes.
 */
export function translateDocument(): void {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (key) {
      el.textContent = i18next.t(key);
    }
  });

  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (key) {
      el.setAttribute("placeholder", i18next.t(key));
    }
  });

  document.querySelectorAll("[data-i18n-aria-label]").forEach((el) => {
    const key = el.getAttribute("data-i18n-aria-label");
    if (key) {
      el.setAttribute("aria-label", i18next.t(key));
    }
  });

  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    const key = el.getAttribute("data-i18n-title");
    if (key) {
      el.setAttribute("title", i18next.t(key));
    }
  });
}

/**
 * Sets the application language, updates the document direction (LTR/RTL), and translates the UI.
 * @param {string} lang The language code to set (e.g., 'en', 'ja', 'ar', 'he').
 * @returns {Promise<void>}
 */
export async function setLanguage(lang: string): Promise<void> {
  localStorage.setItem("app-lang", lang);
  await i18next.changeLanguage(lang);
  document.documentElement.lang = lang;
  setDocumentDir(lang);
  translateDocument();
}

export default i18next;
