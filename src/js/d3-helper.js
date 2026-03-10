/**
 * D3知识图谱可视化帮助函数库
 * 用于提供复用的知识图谱可视化功能
 */

// 此模块依赖d3
import * as d3 from 'd3'; 

/**
 * 创建基础力导向图可视化
 * @param {Object} container - 容器DOM元素或选择器
 * @param {Object} data - 图谱数据 {nodes: [], links: []}
 * @param {Object} options - 可选配置项
 * @returns {Object} - 包含图谱实例和方法的对象
 */
export function createForceGraph(container, data, options = {}) {
  const defaults = {
    width: 800,
    height: 600,
    nodeSize: 15,
    linkDistance: 100,
    chargeStrength: -150,
    colorScheme: d3.schemeCategory10
  };
  
  const config = {...defaults, ...options};
  
  // 创建SVG元素
  const svg = d3.select(container)
    .append("svg")
    .attr("width", config.width)
    .attr("height", config.height);
    
  // 创建力模拟
  const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id(d => d.id).distance(config.linkDistance))
    .force("charge", d3.forceManyBody().strength(config.chargeStrength))
    .force("center", d3.forceCenter(config.width / 2, config.height / 2))
    .force("collide", d3.forceCollide().radius(d => config.nodeSize + 5));
    
  // 创建连接线
  const link = svg.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(data.links)
    .enter()
    .append("line")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .attr("stroke-width", d => Math.sqrt(d.value || 1));
    
  // 创建节点
  const node = svg.append("g")
    .attr("class", "nodes")
    .selectAll("circle")
    .data(data.nodes)
    .enter()
    .append("circle")
    .attr("r", config.nodeSize)
    .attr("fill", (d, i) => config.colorScheme[i % config.colorScheme.length])
    .call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended));
      
  // 添加节点标签
  const label = svg.append("g")
    .attr("class", "labels")
    .selectAll("text")
    .data(data.nodes)
    .enter()
    .append("text")
    .text(d => d.name || d.id)
    .attr("dx", config.nodeSize + 5)
    .attr("dy", ".35em")
    .style("font-size", "12px");
    
  // 更新力模拟时的位置
  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
      
    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
      
    label
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });
  
  // 拖拽行为函数
  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  
  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  
  // 返回图谱实例和操作方法
  return {
    svg,
    simulation,
    updateData: function(newData) {
      // 更新数据和重启模拟的逻辑
      simulation.nodes(newData.nodes);
      simulation.force("link").links(newData.links);
      simulation.alpha(1).restart();
      
      // 更新DOM元素（此处简化，实际应使用D3的enter/exit/update模式）
      node.data(newData.nodes);
      link.data(newData.links);
      label.data(newData.nodes).text(d => d.name || d.id);
    },
    resize: function(width, height) {
      // 调整SVG大小和力模拟中心
      svg.attr("width", width).attr("height", height);
      simulation.force("center", d3.forceCenter(width / 2, height / 2));
      simulation.alpha(0.3).restart();
    }
  };
} 