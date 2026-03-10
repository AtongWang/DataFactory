const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: {
    main: './src/js/main.js',
    // 可以根据需要添加其他入口点
  },
  output: {
    filename: 'js/[name].js',
    path: path.resolve(__dirname, 'static'),
    clean: false, // 不清除未由webpack生成的文件
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.(woff|woff2|eot|ttf|otf|svg)(\?v=\d+\.\d+\.\d+)?$/i,
        type: 'asset/resource',
        generator: {
          filename: 'webfonts/[name][ext]'
        }
      },
    ],
  },
  plugins: [
    // 复制需要的静态资源到目标目录
    new CopyPlugin({
      patterns: [
        { 
          from: "node_modules/bootstrap/dist/css/bootstrap.min.css", 
          to: "css/vendor/bootstrap.min.css" 
        },
        { 
          from: "node_modules/bootstrap/dist/js/bootstrap.bundle.min.js", 
          to: "js/vendor/bootstrap.bundle.min.js" 
        },
        { 
          from: "node_modules/jquery/dist/jquery.min.js", 
          to: "js/vendor/jquery.min.js" 
        },
        {
          from: "node_modules/d3/dist/d3.min.js",
          to: "js/vendor/d3.min.js"
        },
        {
          from: "node_modules/marked/marked.min.js",
          to: "js/vendor/marked.min.js"
        },
        {
          from: "node_modules/dompurify/dist/purify.min.js",
          to: "js/vendor/purify.min.js"
        }
      ],
    }),
  ],
}; 
