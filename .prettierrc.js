export default {
  printWidth: 180, // 换行的行长度
  tabWidth: 2, // 每个缩进级别的空格数
  useTabs: false, // 使用制表符(tab)而不是空格缩进
  semi: true, // 在语句末尾打印分号
  singleQuote: true, // 使用单引号而不是双引号
  quoteProps: "as-needed", // 对象属性引号的使用["as-needed"-仅在必要时加引号;"consistent"-如果一个属性需要引号，则所有属性都加引号;"preserve"-保留输入中的引号使用]
  jsxSingleQuote: false, // JSX中使用单引号而不是双引号
  trailingComma: "es5", // ES5中有效的尾随逗号["es5"-ES5中有效的逗号(对象、数组等);"none"-无尾随逗号;"all"-尽可能使用尾随逗符(包括函数参数和调用)],
  bracketSpacing: true, // 对象字面量中括号之间的空格
  bracketSameLine: true, // HTML/JSX/Vue/Angular元素的>放在最后一行的末尾而不是单独一行
  arrowParens: "always", // 箭头函数参数周围的括号，
};
