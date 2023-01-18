% Author: Henry Ji
% Email: jeep.jipu@gmail.com

% 表格: https://www.tablesgenerator.com/
% 公式: https://mathpix.com/
% 字数统计: https://docwordcounter.com/app/wordcountprocess

% NOTE: 需要pygments依赖(渲染工具)

\ProvidesClass{morningstar}[2022/01/15 Morningstar class]
\NeedsTeXFormat{LaTeX2e}

\RequirePackage{kvoptions}  %提供键值可选项
\RequirePackage{etoolbox}  %提供前端工具
\RequirePackage{calc}  %提供一些参数的符号设置

\SetupKeyvalOptions{family=MORNINGSTAR, prefix=MORNINGSTAR@, setkeys=\kvsetkeys}
\newcommand{\ekv}[1]{\kvsetkeys{MORNINGSTAR}{#1}}

\LoadClass[a4paper,twoside,fontset=none,sub3section]{ctexart}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ctexart}}


%% 参考文献
\RequirePackage{gbt7714}
\citestyle{numbers}  %使用正文模式
\bibliographystyle{gbt7714-numerical}
\renewcommand\refname{参考文献}
\newcommand{\addbib}[1]{
  \setstretch{1}
  \bibliography{#1}
  \setstretch{1.5}
}


%% 数学
\RequirePackage{amsthm}  % 定理
\RequirePackage{amsmath}  % 公式
\numberwithin{equation}{section}  % 按照章节标号
\RequirePackage{mathtools}  % 对amsmath的封装
\RequirePackage{siunitx}  % 单位
\RequirePackage{amssymb}  % 符号
\RequirePackage{braket}  % 内积括号
\renewcommand{\theequation}{\arabic{section}-\arabic{equation}}

\theoremstyle{plain}
\newtheorem{theorem}{定理}[section]
\newtheorem{lemma}[theorem]{引理}
\newtheorem{proposition}[theorem]{命题}
\newtheorem*{corollary}{推论}
\theoremstyle{definition}
\newtheorem{definition}{定义}[section]
\newtheorem{conjecture}{猜想}[section]
\newtheorem{example}{例}[section]
\theoremstyle{remark}
\newtheorem*{remark}{\normalfont\bfseries 评论}
\newtheorem*{note}{\normalfont\bfseries 注}
\newtheorem{case}{\normalfont\bfseries 案例}
\renewcommand*{\proofname}{\normalfont\bfseries 证明}


%% 化学式
\RequirePackage{chemformula}


%% 代码
\RequirePackage{minted}  % NOTE: 需要Pygments环境


%% 算法
\RequirePackage[linesnumbered,ruled,vlined]{algorithm2e}


%% 表格
\RequirePackage{multirow}  % 多行
\RequirePackage{multicol}  % 多列
\RequirePackage{makecell} % 单元格处理
\RequirePackage{longtable}
\RequirePackage{xtab}  % 长表格
\RequirePackage{xltabular}  % 定宽表格
\RequirePackage{booktabs}  % 书籍表格
\RequirePackage{diagbox}
\RequirePackage{arydshln}  % 表线控制
\RequirePackage{array}  % 表列格式
\RequirePackage{tabu}  % 综合应用

\renewcommand{\arraystretch}{1.5}  % 避免表格线与文字靠得太近

% 用于表格中的粗线
\newcommand{\thickhline}{%
    \noalign {\ifnum 0=`}\fi \hrule height 1pt
    \futurelet \reserved@a \@xhline
}

\renewcommand\tablename{表}


%% 插图
\RequirePackage{pdfpages}  % 插入PDF
\RequirePackage{graphicx}
\graphicspath{{image/}{figure/}{fig/}{img/}}
\RequirePackage{tikz} % latex绘图

\renewcommand\figurename{图}


%% 修改图、表的标号
\usepackage{chngcntr}
\counterwithin{table}{section}
\counterwithin{figure}{section}


%% 图表标题修改
\RequirePackage{caption}
\DeclareCaptionLabelSeparator{enskip}{\enskip}
\captionsetup{font=small,labelsep=enskip}
\captionsetup[table]{skip=3pt}
\captionsetup[figure]{skip=3pt}

\newcommand\figref[1]{{\bfseries 图~\ref{#1}}}
\newcommand\tabref[1]{{\bfseries 表~\ref{#1}}}


%% 浮动体
\RequirePackage{float} % 其它环境: float, table
\RequirePackage{caption}  % 浮动标题


%% 自动化工具
\RequirePackage{hyperref}  % 链接与书签
\hypersetup{
  pdfborder={0 0 0},
  colorlinks=true,
  linkcolor={dark},
  urlcolor={black},
  filecolor={winered},
  citecolor={winered},
  linktoc=all,
}
\RequirePackage{bookmark}  % 用于自定义相关属性


%% 颜色
\RequirePackage{xcolor}  % 为code和hyperref提供颜色
\definecolor{winered}{rgb}{0.5,0,0}
\definecolor{dark}{rgb}{0,0,0}
\definecolor{lightgrey}{rgb}{0.95,0.95,0.95}
\definecolor{commentcolor}{RGB}{0,100,0}
\definecolor{frenchplum}{RGB}{190,20,83}
\definecolor{fancycolor}{rgb}{0.5,0.5,0.5}


%% 版面布局
\RequirePackage[
  left=1in,
  right=1in,
  top=1.1in,
  bottom=1.2in,
  % headheight=9pt,
  headsep=20pt,
]{geometry}

\RequirePackage{fancyhdr}  % 设置页眉页脚
\pagestyle{fancy}  % 设置页眉 
\setlength{\headheight}{1.5em}  % 设置页眉高度
\fancyhf{}  % 清空
\fancyhead[EC]{\textcolor{fancycolor}{\zihao{-5}\leftmark}}
\fancyhead[OC]{\textcolor{fancycolor}{\zihao{-5} \@purpose }}
\fancyfoot[C]{\textcolor{fancycolor}{\zihao{-5}\thepage}}


%% 列表环境定制
\RequirePackage[shortlabels,inline]{enumitem}
\setlist{nosep}  % 适配汉字不加过多间隙
\setlist{nolistsep}


%% 字体
\RequirePackage{anyfontsize}  % 灵活设置字体大小
\setmainfont{Times New Roman}
\setmonofont{Source Code Pro}
\setCJKmainfont[AutoFakeSlant,AutoFakeBold]{SimSun}
\setCJKmonofont[AutoFakeSlant,AutoFakeBold]{KaiTi}
\newcommand{\coverTitleFont}{\CJKfontspec[Path=asset/font/]{STXINWEI.ttf}}


%% 间距
% 行距
\RequirePackage{setspace}
\AfterEndPreamble{
  \setstretch{1.5}
}
% 段落
\RequirePackage{indentfirst}  % 段落首行缩进
\setlength{\parindent}{2em}
\setlength{\parskip}{0.5em}
% 特定
\newcommand\halflinespace{\vspace{0.5em}} % 半行


%% 信息
\let\@title\@empty
\let\@entitle\@empty
\let\@major\@empty
\let\@class\@empty
\let\@phone\@empty
\let\@id\@empty
\let\@author\@empty
\let\@instructor\@empty
\let\@period\@empty
\let\@date\@empty
\let\@grade\@empty
\let\@purpose\@empty
\DeclareRobustCommand{\title}[1]{\gdef\@title{#1}}
\DeclareRobustCommand{\entitle}[1]{\gdef\@entitle{#1}}
\DeclareRobustCommand{\major}[1]{\gdef\@major{#1}}
\DeclareRobustCommand{\class}[1]{\gdef\@class{#1}}
\DeclareRobustCommand{\phone}[1]{\gdef\@phone{#1}}
\DeclareRobustCommand{\id}[1]{\gdef\@id{#1}}
\DeclareRobustCommand{\author}[1]{\gdef\@author{#1}}
\DeclareRobustCommand{\instructor}[1]{\gdef\@instructor{#1}}
\DeclareRobustCommand{\period}[1]{\gdef\@period{#1}}
\DeclareRobustCommand{\date}[1]{\gdef\@date{#1}}
\DeclareRobustCommand*{\grade}[1]{\gdef\@grade{#1}}
\DeclareRobustCommand{\purpose}[1]{\gdef\@purpose{#1}}


%% 封面
\newcommand\makecover{%
  \cleardoublepage
  \thispagestyle{empty}
  \begin{flushright}
    \begingroup
    \let\small\relax
    {\zihao{4}
    \def\arraystretch{0.8}
    \begin{tabular}{rc}
      \textbf{学\hfill 号} & \@id     \\ \cline{2-2}
      \textbf{年\hfill 级} & \@grade  \\ \cline{2-2}
    \end{tabular}
    \par}
  \endgroup
  \end{flushright}
  \begin{center}
    \includegraphics[width=1.0\textwidth]{asset/img/河海大学.jpg}
    \vskip 3em
    {\zihao{-0}\coverTitleFont \@purpose \par}
    \vskip 10em
    \begingroup
      \let\small\relax
      {\zihao{3}
      \def\arraystretch{1.0}
      \begin{tabular}{r@{:}c}
        \textbf{专\hfill 业} & \@major       \\ \cline{2-2}
        \textbf{姓\hfill 名} & \@author      \\ \cline{2-2}
        \textbf{联系电话}     & \@phone       \\ \cline{2-2}
        \textbf{指导教师}     & \makebox[160pt]{\@instructor}  \\ \cline{2-2}
      \end{tabular}
      \par}
    \endgroup
    \vskip \stretch{1}
    {\zihao{3} \textbf{\@date} \par}
  \end{center}
  \clearpage
}


%% 邮箱
\newcommand\email[1]{\href{mailto:#1}{\nolinkurl{#1}}}


%% 显示文章标题
\newcommand\showcntitle{
  \vspace*{1cm}
  \centerline{\textbf{\zihao{-1}\@title}}
}
\newcommand\showentitle{
  \vspace*{1cm}
  \begin{center}
  \textbf{\zihao{-1}\@entitle}
  \end{center}
}


%% 摘要
% 显示摘要中的关键词
\RequirePackage{abstract}
\renewcommand\abstractname{\zihao{-2} 摘\hspace{2em}要}
\newcommand{\keywords}[1]{\vskip2ex\par\noindent\normalfont{\bfseries 关键词: } #1}
\newcommand{\showcnkeywords}[1]{
  \noindent \textbf{关键词: } #1
}
\newcommand{\showenkeywords}[1]{
  \noindent \textbf{Key Words: } #1
}


%% ctex配置
\ctexset{
  section={
    format += \zihao{2},
    name = {第,章},
    % format = \bfseries\raggedright\zihao{2},
    % name = {,、},
    % aftername = ,
    number = \chinese{section},
    beforeskip = 1.0ex plus 0.2ex minus .2ex,
		afterskip = 1.0ex plus 0.2ex minus .2ex,
  },
  subsection={
    format += \zihao{-2} \raggedright ,
  },
  subsubsection={
    format += \zihao{3} \raggedright ,
  },
  paragraph={
    format += $\blacksquare~$ \zihao{4},
    beforeskip = 0ex plus 0.1ex minus .1ex,
    afterskip = 0ex plus 0.1ex minus .1ex,
  },
  subparagraph={
    format += $\Diamond~$ \zihao{4},
  },
}


%% 目录
\setcounter{tocdepth}{3}  % 设置在 ToC 的显示的章节深度
\setcounter{secnumdepth}{3}  % 设置章节的编号深度
\RequirePackage[titles]{tocloft}
\renewcommand\cftdot{.}
\renewcommand\cftsecleader{\normalfont\cftdotfill{\cftdotsep}}

\renewcommand\contentsname{目录}


%% 附录
\RequirePackage{appendix}  % 附录宏包
\RequirePackage{tocvsec2}  % 可以随处定义加入目录的级别
\newcommand\startappendix{  % 开始附录
  \clearpage
  \appendix
  \addappheadtotoc
  \appendixpage
  \settocdepth{part}  % 排除part以下内容出目录
}
\renewcommand{\appendixtocname}{附录}
\renewcommand{\appendixpagename}{附录}


%% 正文全局字体
\newcommand\setfontformat{\zihao{-4}}


%% 中文的强调
\renewcommand\emph[1]{
  \textbf{#1}
}


%% 页码标记
\newcommand\startfrontpart{
  \pagenumbering{Roman}  % 罗马数字页码
}
\newcommand\startmainpart{
  \pagenumbering{arabic}  % 阿拉伯数字页码
}


%% 特定功能的章节
% section 章节不标号，添加到目录
\newcommand\sectionWithoutNumbering[1]{
\section*{#1}
\addcontentsline{toc}{section}{#1}
\markboth{#1}{#1}
}
% subsubsection章节标号，不添加到目录
\newcommand\hiddensubsubsection[1]{
  \stepcounter{subsubsection}
  \noindent \textbf{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}\hspace{1em}#1}
}


%% 其他
\RequirePackage{lipsum}  % lorem ipsum
\RequirePackage{fancyvrb}  % 一系列环境配置
\RequirePackage{hologo}  % 设置类似latex那样的logo(特殊字) 
\RequirePackage{verbatim}  % comment环境
\newcommand\labeledCaption[1]{
  \caption{#1}
  \label{#1}
}
\newcommand\versiontext{版本：}
\newcommand\updatetext{日期：}