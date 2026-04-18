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
    chatNumber: "Chat #{{count}}",
    temporaryChat: "Temporary chat",
    copyOf: "{{title}} (Copy)",
    newChat: "+ New Chat",
    schemaExplorer: "Schema Explorer",
    loadingSchema: "Loading schema...",
    model: "Model:",
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
    editChatTitle: "Edit chat title",
    duplicateChat: "Duplicate chat",
    deleteChat: "Delete chat",
    chatActions: "Chat actions",
    chatOptions: "Chat options",
    codeBlock: "Code block",
    chatInputHelp: "Press Enter to send, Shift+Enter for a new line",
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
  },
};

/**
 * Japanese translation resources.
 */
export const jaTranslations = {
  app: {
    title: "t1d-analytics",
    chatNumber: "チャット #{{count}}",
    temporaryChat: "一時的なチャット",
    copyOf: "{{title}} (コピー)",
    newChat: "+ 新しいチャット",
    schemaExplorer: "スキーマエクスプローラー",
    loadingSchema: "スキーマを読み込み中...",
    model: "モデル:",
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
    editChatTitle: "チャットタイトルを編集",
    duplicateChat: "チャットを複製",
    deleteChat: "チャットを削除",
    chatActions: "チャットアクション",
    chatOptions: "チャットオプション",
    codeBlock: "コードブロック",
    chatInputHelp: "Enterで送信、Shift+Enterで改行",
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
  },
};

/**
 * Arabic translation resources.
 */
export const arTranslations = {
  app: {
    title: "t1d-analytics",
    chatNumber: "الدردشة #{{count}}",
    temporaryChat: "دردشة مؤقتة",
    copyOf: "{{title}} (نسخة)",
    newChat: "+ دردشة جديدة",
    schemaExplorer: "مستكشف المخطط",
    loadingSchema: "جاري تحميل المخطط...",
    model: "النموذج:",
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
    editChatTitle: "تعديل عنوان الدردشة",
    duplicateChat: "تكرار الدردشة",
    deleteChat: "حذف الدردشة",
    chatActions: "إجراءات الدردشة",
    chatOptions: "خيارات الدردشة",
    codeBlock: "كتلة التعليمات البرمجية",
    chatInputHelp: "اضغط على Enter للإرسال، Shift+Enter لسطر جديد",
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
  },
};

/**
 * Hebrew translation resources.
 */
export const heTranslations = {
  app: {
    title: "t1d-analytics",
    chatNumber: "צ'אט #{{count}}",
    temporaryChat: "צ'אט זמני",
    copyOf: "{{title}} (עותק)",
    newChat: "+ צ'אט חדש",
    schemaExplorer: "סייר סכמות",
    loadingSchema: "טוען סכמה...",
    model: "מודל:",
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
    editChatTitle: "ערוך כותרת צ'אט",
    duplicateChat: "שכפל צ'אט",
    deleteChat: "מחק צ'אט",
    chatActions: "פעולות צ'אט",
    chatOptions: "אפשרויות צ'אט",
    codeBlock: "בלוק קוד",
    chatInputHelp: "לחץ Enter לשליחה, Shift+Enter לשורה חדשה",
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
  },
};

i18next.init({
  lng: "en",
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

export async function setLanguage(lang: string): Promise<void> {
  await i18next.changeLanguage(lang);
  document.documentElement.lang = lang;
  document.documentElement.dir = ["ar", "he"].includes(lang) ? "rtl" : "ltr";
  translateDocument();
}

export default i18next;
